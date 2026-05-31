//! Decoder-output SHA-256 pin for the two 1920×1080 interlaced fixtures
//! under `docs/video/prores/fixtures/`.
//!
//! RDD 36 §5.1 + §6.2 Table 2 + §7.5.3 govern the interlaced path:
//! a single 1920×1080 `frame_size` container holds TWO picture headers
//! (one per field) and the decoder re-interleaves the field rows on
//! output (top field → even rows for `interlace_mode = 1`, bottom →
//! odd rows; swapped for mode 2). The visible per-frame yuv422p10le
//! byte budget is exactly `(1920·1080 + 2·960·1080)·2 = 8 294 400`
//! bytes.
//!
//! What this test pins
//! --------------------
//! Both fixtures (`pal-1080i50` and `interlaced-tff`) carry a
//! byte-identical apcn elementary stream — the only difference between
//! them is the surrounding MOV `moov` metadata. The decoder MUST
//! therefore produce identical YUV output across the two, which itself
//! is a real check (it catches regressions where the decoder picks up
//! anything outside the elementary frame from the MOV wrapper, or
//! where field ordering is sensitive to surrounding container state).
//!
//! Beyond the cross-fixture identity, this test pins a SHA-256 of the
//! decoded byte stream so a future regression to either the §7.5.3
//! field-row mapping, the §7.4 IDCT scaling, the §5.3 slice walker, or
//! the §5.1 multi-picture frame parser flips the test red instead of
//! silently shifting decode output. The pinned SHA reflects the
//! in-tree Annex A-compliant float IDCT; it is **not** the
//! fixed-point IDCT SHA shipped in each fixture's
//! `expected.yuv.sha256` (~1-LSB divergence per RDD 36 §7.4 is
//! permitted between any two Annex A-compliant IDCT formulations).
//! The reference SHA is read from the manifest and reported in the
//! test log alongside ours so the gap is visible.
//!
//! The fixture's `expected.yuv.sha256` covers the **first frame only**
//! (`size=8294400`); we score only frame 0 against the pin to match
//! that scope.
//!
//! Clean-room note: no external decoder source consulted. The fixture
//! manifests + RDD 36 §5.1 / §6.2 / §7.5.3 are the only references.

use std::fs;
use std::path::PathBuf;

use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

const W: usize = 1920;
const H: usize = 1080;
/// `(W·H + 2·(W/2)·H) · 2 bytes` for yuv422p10le.
const FRAME_BYTES: usize = (W * H + 2 * (W / 2) * H) * 2;

/// SHA-256 of the decoded first frame in Y/Cb/Cr planar order
/// (yuv422p10le, LE-packed), as produced by the in-tree float-IDCT
/// decoder on the two 1920×1080 interlaced fixtures. Both fixtures
/// share the same elementary bitstream, so the SHA MUST match across
/// them.
///
/// Update procedure when the IDCT path legitimately changes: run the
/// test, copy the printed `OUR_SHA256` value, replace the constant
/// here, document the change in CHANGELOG.md. Drift outside of an
/// intentional IDCT switch is a real regression and must be
/// investigated.
const OUR_FRAME0_SHA256_HEX: &str =
    "101d1a9bc8e467fa2cb2b00a87cf7d535617c48bd8278fc89a0d05d1ba8ca736";

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

// ---------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4). Tests-side only, zero external deps —
// it's a generic hash, not part of the codec surface. Adding a
// `sha2` dev-dependency would also be defensible but keeping the test
// binary minimal avoids any new transitive deps for downstream
// `cargo test` users.
// ---------------------------------------------------------------------

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
    // Padded message length = data + 0x80 + zeros + 8-byte length,
    // padded to a multiple of 64. Compute zero count without
    // materialising a copy of the input.
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

/// SHA-256 self-check against the FIPS 180-4 §B.1 vectors. Catches
/// any future typo in the K constants / round equations above before
/// it can mask a real decoder regression.
#[test]
fn sha256_self_check_against_fips_180_4_vectors() {
    // §B.1 — empty input.
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    // §B.1 — "abc".
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // §B.2 — 448-bit input ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").
    assert_eq!(
        hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

/// Decode frame 0 of the given fixture (`pal-1080i50` /
/// `interlaced-tff`) as `yuv422p10le` and return the concatenated
/// Y/Cb/Cr plane bytes, or `None` when the fixture file is missing
/// (standalone crate checkout without the docs submodule).
fn decode_first_frame(name: &str) -> Option<Vec<u8>> {
    let in_path = fixture_dir(name).join("input.mov");
    let bytes = match fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {name}: missing {} ({e}). docs/ fixtures live in the \
                 workspace umbrella repo — the standalone crate checkout has \
                 no corpus.",
                in_path.display()
            );
            return None;
        }
    };
    let frames = extract_prores_frames(&bytes);
    assert!(
        !frames.is_empty(),
        "{name}: container parser found no `icpf` frames in {} bytes",
        bytes.len()
    );
    let decoded = decode_packet_with_depth(
        &frames[0],
        Some(0),
        Some((BitDepth::Ten, ChromaFormat::Y422)),
    )
    .unwrap_or_else(|e| panic!("{name} frame 0 decode failed: {e:?}"));
    assert_eq!(
        decoded.planes.len(),
        3,
        "{name}: expected 3 yuv422p10le planes, got {}",
        decoded.planes.len()
    );
    let mut out = Vec::with_capacity(FRAME_BYTES);
    for p in &decoded.planes {
        out.extend_from_slice(&p.data);
    }
    assert_eq!(
        out.len(),
        FRAME_BYTES,
        "{name}: decoded frame 0 has {} bytes, expected {FRAME_BYTES} for 1920x1080 yuv422p10le",
        out.len()
    );
    Some(out)
}

/// Parse the first 64 hex chars of an `expected.yuv.sha256` manifest
/// line. The format the docs/ collaborator ships is
/// `<hex>  expected.yuv  size=<n>  pix_fmt=yuv422p10le` (the trailing
/// fields are advisory). Returns `None` when the manifest file is
/// missing or unparseable.
fn read_reference_sha_hex(name: &str) -> Option<String> {
    let path = fixture_dir(name).join("expected.yuv.sha256");
    let text = fs::read_to_string(&path).ok()?;
    let first_token = text.split_whitespace().next()?;
    if first_token.len() < 64 || !first_token.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(first_token[..64].to_ascii_lowercase())
}

/// RDD 36 §5.1 / §7.5.3 — interlaced TFF 1920×1080 apcn frame 0
/// decode pin. Hashes the Y/Cb/Cr byte stream as the decoder emits it
/// and asserts the SHA matches the pinned value. Updating the pin
/// requires a CHANGELOG entry; drift outside an intentional IDCT
/// switch is a real regression.
#[test]
fn interlaced_tff_frame0_sha_pin() {
    let Some(bytes) = decode_first_frame("interlaced-tff") else {
        return;
    };
    let our = hex(&sha256(&bytes));
    let reference = read_reference_sha_hex("interlaced-tff");
    eprintln!(
        "interlaced-tff frame 0: OUR_SHA256={our} REFERENCE_SHA256={}",
        reference.as_deref().unwrap_or("<missing>")
    );
    assert_eq!(
        our, OUR_FRAME0_SHA256_HEX,
        "interlaced-tff frame 0 SHA drifted — review the §5.1 multi-picture \
         walker, §7.5.3 field-row deinterleave, §5.3 slice walker, and §7.4 \
         IDCT path. Update OUR_FRAME0_SHA256_HEX + CHANGELOG only if the \
         change is an intentional IDCT switch."
    );
}

/// RDD 36 §5.1 / §7.5.3 — PAL 1080i50 broadcast apcn frame 0 decode
/// pin. Same elementary bitstream as `interlaced-tff` (see fixture
/// `notes.md`), so the decoder MUST produce identical bytes — a
/// mismatch points to container state leaking into the elementary
/// decoder.
#[test]
fn pal_1080i50_frame0_sha_pin() {
    let Some(bytes) = decode_first_frame("pal-1080i50") else {
        return;
    };
    let our = hex(&sha256(&bytes));
    let reference = read_reference_sha_hex("pal-1080i50");
    eprintln!(
        "pal-1080i50 frame 0: OUR_SHA256={our} REFERENCE_SHA256={}",
        reference.as_deref().unwrap_or("<missing>")
    );
    assert_eq!(
        our, OUR_FRAME0_SHA256_HEX,
        "pal-1080i50 frame 0 SHA drifted — review the §5.1 multi-picture \
         walker, §7.5.3 field-row deinterleave, §5.3 slice walker, and §7.4 \
         IDCT path. Update OUR_FRAME0_SHA256_HEX + CHANGELOG only if the \
         change is an intentional IDCT switch."
    );
}

/// RDD 36 §5.1 cross-fixture identity check — `pal-1080i50` and
/// `interlaced-tff` ship the same elementary apcn bitstream wrapped in
/// different MOV `moov` metadata. The decoder MUST therefore yield
/// byte-identical YUV. Catches regressions where MOV-side state
/// (display matrix, edit list, sample-description ordering) leaks
/// into the inner ProRes decoder.
#[test]
fn interlaced_tff_and_pal_decode_identically() {
    let Some(a) = decode_first_frame("interlaced-tff") else {
        return;
    };
    let Some(b) = decode_first_frame("pal-1080i50") else {
        return;
    };
    assert_eq!(
        a.len(),
        b.len(),
        "fixture frames differ in size: {} vs {}",
        a.len(),
        b.len()
    );
    if a != b {
        // Spell out the first divergence rather than dumping 8 MB.
        let first_diff = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap();
        panic!(
            "interlaced-tff and pal-1080i50 decode to different bytes; first \
             divergence at byte offset {first_diff}: {:#04x} vs {:#04x}",
            a[first_diff], b[first_diff]
        );
    }
}
