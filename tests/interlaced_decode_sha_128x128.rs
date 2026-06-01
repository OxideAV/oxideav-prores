//! Decoder-output SHA-256 pin for the in-tree small interlaced apcn
//! fixture `docs/video/prores/fixtures/interlaced-tff-128x128-apcn/`.
//!
//! RDD 36 §5.1 + §6.2 Table 2 + §7.5.3 govern the interlaced path:
//! each container frame carries TWO picture headers (one per field)
//! and the decoder re-interleaves the field rows on output. The
//! visible per-frame `yuv422p10le` byte budget is
//! `(128·128 + 2·64·128)·2 = 65 536` bytes.
//!
//! What this test pins
//! --------------------
//! This is the SHA-256 companion to the existing per-field PSNR test
//! `tests/docs_corpus.rs::corpus_interlaced_tff_128x128_apcn_per_field_psnr`.
//! The 128×128 fixture differs from the 1920×1080 broadcast pair
//! (`pal-1080i50` and `interlaced-tff`, pinned in
//! `tests/interlaced_decode_sha.rs`) in three useful ways:
//!
//! 1. **Both frames** of the 2-frame container are pinned, not just
//!    frame 0 — exercises the §5.1 container walker on a SECOND
//!    `icpf`-prefixed frame and the §7.5.3 field-row mapping on a
//!    frame whose synthetic source content differs from frame 0 (so
//!    the per-frame SHAs do NOT collide).
//! 2. **Reference `expected.yuv` is committed in-tree**, alongside
//!    `expected.yuv.sha256` — the manifest's reference SHA covers the
//!    whole 131 072-byte file (size of two frames concatenated), so
//!    we can also pin the **concatenated** decoded SHA and report
//!    the reference SHA alongside ours. The 1920×1080 pair has only
//!    the SHA (no raw YUV), so the existing test only reports per-
//!    frame SHA without a corresponding per-frame reference.
//! 3. The fixture is small enough (65 536 bytes/frame) that the test
//!    can pin **both** frames without bloating the test binary.
//!
//! As with the 1920×1080 pin: the pinned SHA is the in-tree
//! Annex A-compliant **float** IDCT output, NOT the fixed-point IDCT
//! SHA the manifest ships. RDD 36 §7.4 permits ~1 LSB divergence
//! between any two Annex A-compliant IDCT formulations; the
//! float vs fixed-point gap is reported in the test log so the
//! divergence stays visible.
//!
//! Clean-room note: no external decoder source consulted. The
//! fixture manifest + RDD 36 §5.1 / §6.2 / §7.5.3 are the only
//! references.

use std::fs;
use std::path::PathBuf;

use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

const W: usize = 128;
const H: usize = 128;
/// `(W·H + 2·(W/2)·H) · 2 bytes` for yuv422p10le.
const FRAME_BYTES: usize = (W * H + 2 * (W / 2) * H) * 2;

/// SHA-256 of frame 0 of the 128×128 TFF apcn fixture, decoded as
/// `yuv422p10le` by the in-tree float-IDCT decoder. The pinned value
/// reflects the float IDCT, NOT the fixed-point reference SHA shipped
/// alongside in `expected.yuv.sha256` — ~1-LSB divergence is permitted
/// by RDD 36 §7.4 between any two Annex A-compliant IDCT
/// formulations.
///
/// Update procedure when the IDCT path legitimately changes: run the
/// test, copy the printed `OUR_SHA256` value, replace the constant,
/// document the change in CHANGELOG.md. Drift outside of an
/// intentional IDCT switch is a real regression.
const OUR_FRAME0_SHA256_HEX: &str =
    "b5190ef115a970eccf004967066bc45c09184f4c645cf546c201240a0dbbcfa2";

/// SHA-256 of frame 1 of the same fixture. Frame 1's synthetic source
/// content (the second second of `testsrc=size=128x128:rate=25:duration=2`)
/// differs from frame 0, so the per-frame SHAs MUST differ; a regression
/// where the §5.1 walker silently re-decodes frame 0 twice would surface
/// here as `frame1_sha == frame0_sha`.
const OUR_FRAME1_SHA256_HEX: &str =
    "e5ca00a7cf72e694854d69dc93ff9dcdfd980fe60bcb5aef4057189d0adf185a";

/// SHA-256 of the concatenated `frame0 ‖ frame1` byte stream. This
/// matches the scope of the fixture's `expected.yuv.sha256` manifest
/// (which covers `size=131072`, both frames). Reported alongside the
/// per-frame pins so the float vs fixed-point IDCT gap is visible.
const OUR_FULL_SHA256_HEX: &str =
    "497af2ba21b1adfbb1eb38039c4563a384cd7eb454ae19752324fee6c32acda3";

fn fixture_dir() -> PathBuf {
    PathBuf::from("../../docs/video/prores/fixtures/interlaced-tff-128x128-apcn")
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
// it's a generic hash, not part of the codec surface. The
// implementation is duplicated from `tests/interlaced_decode_sha.rs` so
// each test binary stays self-contained (cargo test compiles each
// `tests/*.rs` as its own binary; sharing would mean a `tests/common`
// module). The FIPS 180-4 self-check below guards against typos.
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

/// SHA-256 self-check against the FIPS 180-4 §B.1 / §B.2 vectors.
/// Catches any future typo in the K constants / round equations
/// above before it can mask a real decoder regression.
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
    // §B.2 — 448-bit input.
    assert_eq!(
        hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

/// Decode the requested frame index of the 128×128 fixture as
/// `yuv422p10le` and return the concatenated Y/Cb/Cr plane bytes, or
/// `None` when the fixture file is missing (standalone crate
/// checkout without the docs submodule).
fn decode_frame_bytes(idx: usize) -> Option<Vec<u8>> {
    let in_path = fixture_dir().join("input.mov");
    let bytes = match fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip 128x128 SHA pin: missing {} ({e}). docs/ fixtures \
                 live in the workspace umbrella repo — the standalone \
                 crate checkout has no corpus.",
                in_path.display()
            );
            return None;
        }
    };
    let frames = extract_prores_frames(&bytes);
    assert!(
        frames.len() > idx,
        "interlaced-tff-128x128-apcn: needed frame {idx} but container \
         only carries {} `icpf` frames in {} bytes",
        frames.len(),
        bytes.len()
    );
    let decoded = decode_packet_with_depth(
        &frames[idx],
        Some(idx as i64),
        Some((BitDepth::Ten, ChromaFormat::Y422)),
    )
    .unwrap_or_else(|e| panic!("frame {idx} decode failed: {e:?}"));
    assert_eq!(
        decoded.planes.len(),
        3,
        "expected 3 yuv422p10le planes, got {}",
        decoded.planes.len()
    );
    let mut out = Vec::with_capacity(FRAME_BYTES);
    for p in &decoded.planes {
        out.extend_from_slice(&p.data);
    }
    assert_eq!(
        out.len(),
        FRAME_BYTES,
        "decoded frame {idx} has {} bytes, expected {FRAME_BYTES} for 128x128 yuv422p10le",
        out.len()
    );
    Some(out)
}

/// Parse the first 64 hex chars of an `expected.yuv.sha256` manifest
/// line. Format: `<hex>  expected.yuv  size=<n>  pix_fmt=yuv422p10le`.
fn read_reference_sha_hex() -> Option<String> {
    let path = fixture_dir().join("expected.yuv.sha256");
    let text = fs::read_to_string(&path).ok()?;
    let first_token = text.split_whitespace().next()?;
    if first_token.len() < 64 || !first_token.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(first_token[..64].to_ascii_lowercase())
}

/// RDD 36 §5.1 / §7.5.3 — frame 0 of the 128×128 TFF apcn fixture.
/// Pins the decoder's exact Y/Cb/Cr byte output (yuv422p10le) so any
/// regression to the §5.1 multi-picture frame walker, the §7.5.3
/// field-row deinterleave, the §5.3 slice walker, or the §7.4 IDCT
/// path flips the test red instead of silently shifting output.
#[test]
fn interlaced_128x128_tff_frame0_sha_pin() {
    let Some(bytes) = decode_frame_bytes(0) else {
        return;
    };
    let our = hex(&sha256(&bytes));
    eprintln!("interlaced-tff-128x128-apcn frame 0: OUR_SHA256={our}");
    assert_eq!(
        our, OUR_FRAME0_SHA256_HEX,
        "interlaced-tff-128x128-apcn frame 0 SHA drifted — review the \
         §5.1 multi-picture walker, §7.5.3 field-row deinterleave, §5.3 \
         slice walker, and §7.4 IDCT path. Update OUR_FRAME0_SHA256_HEX \
         + CHANGELOG only if the change is an intentional IDCT switch."
    );
}

/// RDD 36 §5.1 / §7.5.3 — frame 1 of the same fixture. The
/// synthetic `testsrc` source's second second differs from the first,
/// so frame 1's decoded SHA MUST differ from frame 0's. A regression
/// where the §5.1 walker silently re-decodes frame 0 twice would
/// surface here as a SHA collision.
#[test]
fn interlaced_128x128_tff_frame1_sha_pin() {
    let Some(bytes) = decode_frame_bytes(1) else {
        return;
    };
    let our = hex(&sha256(&bytes));
    eprintln!("interlaced-tff-128x128-apcn frame 1: OUR_SHA256={our}");
    assert_eq!(
        our, OUR_FRAME1_SHA256_HEX,
        "interlaced-tff-128x128-apcn frame 1 SHA drifted — same review \
         scope as the frame 0 pin. Update OUR_FRAME1_SHA256_HEX + \
         CHANGELOG only if the change is an intentional IDCT switch."
    );
    assert_ne!(
        OUR_FRAME0_SHA256_HEX, OUR_FRAME1_SHA256_HEX,
        "frame 0 and frame 1 SHA pins are equal — frames should differ; \
         this is a guard against accidentally pinning identical hashes."
    );
}

/// RDD 36 §5.1 / §7.5.3 — concatenated `frame0 ‖ frame1` byte stream
/// SHA pin. Matches the scope of the fixture's
/// `expected.yuv.sha256` manifest (covers `size=131072`, both frames).
/// The pinned value is the in-tree float-IDCT SHA; the reference SHA
/// (fixed-point IDCT, ~1-LSB divergence permitted per RDD 36 §7.4) is
/// read from the manifest and reported in the test log alongside ours
/// so the gap stays visible.
#[test]
fn interlaced_128x128_tff_concatenated_sha_pin() {
    let Some(f0) = decode_frame_bytes(0) else {
        return;
    };
    let Some(f1) = decode_frame_bytes(1) else {
        return;
    };
    let mut all = Vec::with_capacity(2 * FRAME_BYTES);
    all.extend_from_slice(&f0);
    all.extend_from_slice(&f1);
    assert_eq!(
        all.len(),
        2 * FRAME_BYTES,
        "concatenated frame bytes should be exactly 2 × FRAME_BYTES"
    );
    let our = hex(&sha256(&all));
    let reference = read_reference_sha_hex();
    eprintln!(
        "interlaced-tff-128x128-apcn concat (frame0 || frame1): \
         OUR_SHA256={our} REFERENCE_SHA256={}",
        reference.as_deref().unwrap_or("<missing>")
    );
    assert_eq!(
        our, OUR_FULL_SHA256_HEX,
        "interlaced-tff-128x128-apcn concatenated SHA drifted — review \
         the §5.1 multi-picture walker (BOTH frames), §7.5.3 field-row \
         deinterleave, §5.3 slice walker, and §7.4 IDCT path. Update \
         OUR_FULL_SHA256_HEX + CHANGELOG only if the change is an \
         intentional IDCT switch."
    );
}
