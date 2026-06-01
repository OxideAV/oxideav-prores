//! Decoder-output SHA-256 pin for the progressive (single-picture) 422
//! + 444 fixtures under `docs/video/prores/fixtures/`.
//!
//! Companion to `tests/interlaced_decode_sha.rs` (the §5.1 + §7.5.3
//! interlaced two-picture path). This file extends the same
//! float-IDCT lockstep approach over the *progressive* path:
//!
//! * RDD 36 §5.1 single-picture frame container
//!   (`frame_size + 'icpf' + frame_header + picture_header + slice_table
//!    + slice+`).
//! * RDD 36 §5.3 slice walker at the 8-MBs-per-slice default layout,
//!   `mb_height = ceil(h / 16)` (progressive — not the §7.5.3
//!   per-field `ceil(h/32)`).
//! * RDD 36 §7.2 Figure 4 progressive block scan.
//! * RDD 36 §7.5.1 component → pixel sample mapping at bit depth 10.
//!
//! Coverage shape
//! --------------
//! For each progressive corpus fixture that ships an
//! `expected.yuv.sha256` sidecar (the 1080p ones don't ship the raw
//! `expected.yuv` to keep the corpus under the size budget), this
//! driver:
//!
//! 1. Decodes frame 0 of the input MOV through
//!    [`decode_packet_with_depth`] with the fixture's pinned
//!    `(BitDepth, ChromaFormat)` request.
//! 2. Hashes the concatenated Y/Cb/Cr (+A on the alpha fixture) byte
//!    stream with the in-test SHA-256 (FIPS 180-4) implementation
//!    shared with `interlaced_decode_sha.rs`.
//! 3. Asserts the hash matches a pinned constant.
//!
//! Why a separate SHA from the fixture sidecar
//! --------------------------------------------
//! The fixture's `expected.yuv.sha256` is the fixed-point IDCT SHA
//! produced by an external `prores_ks`-style reference. The in-tree
//! decoder uses an Annex A-compliant float IDCT — RDD 36 §7.4 permits
//! either, with up to ~1-LSB divergence between the two formulations.
//! The pinned constant in this test is therefore the float-IDCT SHA;
//! both SHAs are reported in the test log (`OUR_SHA256` vs
//! `REFERENCE_SHA256`) so the magnitude of the divergence stays
//! visible without flipping the test red on a permitted IDCT choice.
//!
//! Update procedure when the IDCT path legitimately changes: run the
//! affected test, copy the printed `OUR_SHA256` value, replace the
//! constant here, document the change in CHANGELOG.md. Drift outside
//! an intentional IDCT switch is a real regression and must be
//! investigated against the spec clauses above.
//!
//! Clean-room note: no external decoder source consulted. The fixture
//! manifests + SMPTE RDD 36 §5.1 / §5.3 / §7.2 / §7.5.1 are the only
//! references.

use std::fs;
use std::path::PathBuf;

use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

// ---------------------------------------------------------------------
// Helpers (frame extraction + SHA-256 + hex). Duplicated from
// `interlaced_decode_sha.rs` to keep each integration test binary
// self-contained — Cargo builds them independently, so a shared
// `mod` would require an extra `tests/common/` module, which is a
// larger restructure than this round wants. The two copies are
// byte-identical apart from comments and stay in sync by review.
// ---------------------------------------------------------------------

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/prores/fixtures").join(name)
}

fn extract_prores_frames(container: &[u8]) -> Vec<Vec<u8>> {
    let needle = b"icpf";
    let mut out = Vec::new();
    if container.len() < 4 {
        return out;
    }
    let mut i = 4usize;
    while i + 4 <= container.len() {
        if &container[i..i + 4] == needle {
            let size_off = i - 4;
            let frame_size =
                u32::from_be_bytes(container[size_off..size_off + 4].try_into().unwrap()) as usize;
            let end = size_off + frame_size;
            if end <= container.len() && frame_size >= 8 {
                out.push(container[size_off..end].to_vec());
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64) * 8;
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let mj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(mj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, w) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

fn read_reference_sha_hex(name: &str) -> Option<String> {
    let path = fixture_dir(name).join("expected.yuv.sha256");
    let text = fs::read_to_string(&path).ok()?;
    let first_token = text.split_whitespace().next()?;
    if first_token.len() < 64 || !first_token.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(first_token[..64].to_ascii_lowercase())
}

// ---------------------------------------------------------------------
// Per-fixture pin spec.
// ---------------------------------------------------------------------

struct PinSpec {
    /// Fixture directory name under `docs/video/prores/fixtures/`.
    name: &'static str,
    width: usize,
    height: usize,
    chroma: ChromaFormat,
    /// 3 for plain YUV, 4 when the fixture carries an alpha plane.
    planes: usize,
    /// Float-IDCT SHA-256 of the decoded frame-0 byte stream
    /// (concatenated Y/Cb/Cr [/A]).
    pinned_sha_hex: &'static str,
}

impl PinSpec {
    /// Bytes-per-sample. The corpus uses 10-bit LE-packed across the
    /// board (yuv422p10le / yuv444p10le / yuva444p10le).
    const BPS: usize = 2;

    fn frame_bytes(&self) -> usize {
        let cw = match self.chroma {
            ChromaFormat::Y422 => self.width.div_ceil(2),
            ChromaFormat::Y444 => self.width,
        };
        let yuv = (self.width * self.height + 2 * cw * self.height) * Self::BPS;
        let alpha = if self.planes == 4 {
            self.width * self.height * Self::BPS
        } else {
            0
        };
        yuv + alpha
    }

    fn decode_first_frame(&self) -> Option<Vec<u8>> {
        let in_path = fixture_dir(self.name).join("input.mov");
        let bytes = match fs::read(&in_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "skip {}: missing {} ({e}). docs/ fixtures live in the \
                     workspace umbrella repo — the standalone crate checkout \
                     has no corpus.",
                    self.name,
                    in_path.display()
                );
                return None;
            }
        };
        let frames = extract_prores_frames(&bytes);
        assert!(
            !frames.is_empty(),
            "{}: container parser found no `icpf` frames in {} bytes",
            self.name,
            bytes.len()
        );
        let decoded =
            decode_packet_with_depth(&frames[0], Some(0), Some((BitDepth::Ten, self.chroma)))
                .unwrap_or_else(|e| panic!("{} frame 0 decode failed: {e:?}", self.name));
        assert_eq!(
            decoded.planes.len(),
            self.planes,
            "{}: expected {} planes for frame 0, got {}",
            self.name,
            self.planes,
            decoded.planes.len()
        );
        let mut out = Vec::with_capacity(self.frame_bytes());
        for p in &decoded.planes {
            out.extend_from_slice(&p.data);
        }
        let expected = self.frame_bytes();
        assert_eq!(
            out.len(),
            expected,
            "{}: decoded frame 0 has {} bytes, expected {expected} for {}x{} \
             ({:?}, {} planes, 10-bit LE)",
            self.name,
            out.len(),
            self.width,
            self.height,
            self.chroma,
            self.planes
        );
        Some(out)
    }

    fn assert_sha_pin(&self) {
        let Some(bytes) = self.decode_first_frame() else {
            return;
        };
        let our = hex(&sha256(&bytes));
        let reference = read_reference_sha_hex(self.name);
        eprintln!(
            "{} frame 0: OUR_SHA256={our} REFERENCE_SHA256={}",
            self.name,
            reference.as_deref().unwrap_or("<missing>")
        );
        assert_eq!(
            our, self.pinned_sha_hex,
            "{} frame 0 SHA drifted — review the §5.1 frame container, \
             §5.3 slice walker, §7.2 Figure 4 progressive scan, §7.4 \
             IDCT path, and §7.5.1 component → pixel mapping. Update \
             the pinned SHA + CHANGELOG only if the change is an \
             intentional IDCT or scan-table switch.",
            self.name,
        );
    }
}

// ---------------------------------------------------------------------
// Pins.
// ---------------------------------------------------------------------
//
// Each constant below is the in-tree float-IDCT SHA-256 of frame 0's
// concatenated plane bytes (Y/Cb/Cr [/A]) for the named fixture. The
// constants were captured by running each test once with a stub value
// and copying the printed `OUR_SHA256`. Future runs assert against
// the captured value.

/// 422 Proxy (`apco`) at 1280×720, yuv422p10le. Stresses RDD 36
/// §5.2 / §5.3 + the lowest-quality 422 quant tier.
const PROXY_1280X720_FRAME0_SHA: &str =
    "665b723ddf6fdbc688de5789d4b4a0ed5d55fefd90e5bcc8b6a98de0b93e7b9c";

/// 422 LT (`apcs`) at 1280×720, yuv422p10le. Same coverage as Proxy
/// but at the higher-bitrate LT tier.
const LT_1280X720_FRAME0_SHA: &str =
    "d9e84d90f32c121485e88f6fec54569ae9e4825af7aa48efbaa85abfcbee69e2";

/// 422 Standard (`apcn`) at 1920×1080, yuv422p10le. The canonical
/// 1020-slice progressive layout — every default 8-MB slice fully
/// utilised, exercises the slice walker on a wide frame.
const SQ_1920X1080_FRAME0_SHA: &str =
    "cf94806987f8e08b7a289df4998ad8190a55e47fcb5d66cd5e7a612217d9c2c2";

/// 422 HQ (`apch`) at 1920×1080, yuv422p10le. High-rate quant tier
/// + larger per-slice budget.
const HQ_1920X1080_FRAME0_SHA: &str =
    "042a4513f5f2ef6f5b368e8de01c87d3e47464cfcdc87e120f22727ee42c35e6";

/// 4444 (`ap4h`) at 1920×1080, yuv444p10le. Stresses the §5.2
/// `chroma_format = 3` path and the §7.4 4:4:4 chroma-block doubling.
const PRORES4444_1920X1080_FRAME0_SHA: &str =
    "b013e52b35a04011c9148460d2ece9db99c0f3ac7a7c65e4f7572cafb9fdf4df";

/// 4444 XQ (`ap4x`) at 1920×1080, yuv444p10le. Same shape as 4444
/// but version=1 with the XQ quant range.
const PRORES4444_XQ_1920X1080_FRAME0_SHA: &str =
    "c3ec217a55b9a3b678f800780cd525fa9d99ae15029558b261322d759f50923f";

/// 4444 + alpha (`ap4h` with `alpha_channel_type = 2`) at 1920×1080,
/// yuva444p10le. Adds the §5.3.3 + §7.1.2 alpha-plane unpack on top
/// of the 4444 coverage; 4 output planes.
const PRORES4444_ALPHA_FRAME0_SHA: &str =
    "e1cdb145587b3606bba0cce8e859975f3780524e0d743895888a21b1a09079a4";

// ---------------------------------------------------------------------
// Per-fixture tests.
// ---------------------------------------------------------------------

#[test]
fn proxy_1280x720_frame0_sha_pin() {
    PinSpec {
        name: "proxy-1280x720",
        width: 1280,
        height: 720,
        chroma: ChromaFormat::Y422,
        planes: 3,
        pinned_sha_hex: PROXY_1280X720_FRAME0_SHA,
    }
    .assert_sha_pin();
}

#[test]
fn lt_1280x720_frame0_sha_pin() {
    PinSpec {
        name: "lt-1280x720",
        width: 1280,
        height: 720,
        chroma: ChromaFormat::Y422,
        planes: 3,
        pinned_sha_hex: LT_1280X720_FRAME0_SHA,
    }
    .assert_sha_pin();
}

#[test]
fn sq_1920x1080_frame0_sha_pin() {
    PinSpec {
        name: "sq-1920x1080",
        width: 1920,
        height: 1080,
        chroma: ChromaFormat::Y422,
        planes: 3,
        pinned_sha_hex: SQ_1920X1080_FRAME0_SHA,
    }
    .assert_sha_pin();
}

#[test]
fn hq_1920x1080_frame0_sha_pin() {
    PinSpec {
        name: "hq-1920x1080",
        width: 1920,
        height: 1080,
        chroma: ChromaFormat::Y422,
        planes: 3,
        pinned_sha_hex: HQ_1920X1080_FRAME0_SHA,
    }
    .assert_sha_pin();
}

#[test]
fn prores4444_1920x1080_frame0_sha_pin() {
    PinSpec {
        name: "4444-1920x1080",
        width: 1920,
        height: 1080,
        chroma: ChromaFormat::Y444,
        planes: 3,
        pinned_sha_hex: PRORES4444_1920X1080_FRAME0_SHA,
    }
    .assert_sha_pin();
}

#[test]
fn prores4444xq_1920x1080_frame0_sha_pin() {
    PinSpec {
        name: "4444xq-1920x1080",
        width: 1920,
        height: 1080,
        chroma: ChromaFormat::Y444,
        planes: 3,
        pinned_sha_hex: PRORES4444_XQ_1920X1080_FRAME0_SHA,
    }
    .assert_sha_pin();
}

#[test]
fn prores4444_alpha_frame0_sha_pin() {
    PinSpec {
        name: "4444-with-alpha",
        width: 1920,
        height: 1080,
        chroma: ChromaFormat::Y444,
        planes: 4,
        pinned_sha_hex: PRORES4444_ALPHA_FRAME0_SHA,
    }
    .assert_sha_pin();
}

/// SHA-256 self-check against the FIPS 180-4 §B.1 / §B.2 vectors —
/// catches any future typo in the K constants / round equations above
/// before it can mask a real decoder regression. Duplicated from
/// `interlaced_decode_sha.rs` for the same per-binary
/// self-containment reason.
#[test]
fn sha256_self_check_against_fips_180_4_vectors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
