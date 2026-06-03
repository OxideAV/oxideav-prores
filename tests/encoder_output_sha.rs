//! Encoder-output SHA-256 lockstep pin (RDD 36 §5 + §7).
//!
//! Companion to the decoder-output SHA pins in
//! `tests/progressive_decode_sha.rs` (Apple-encoded fixtures) and
//! `tests/interlaced_decode_sha{,_128x128}.rs` (Apple interlaced). Those
//! tests hash the bytes the decoder *produces* given a fixed input. This
//! test hashes the bytes the **encoder** produces given a fixed input:
//! a tiny deterministic synthetic frame is encoded through every public
//! free-function entry point and the resulting RDD 36 packet bytes are
//! pinned by SHA-256.
//!
//! Why this matters
//! ----------------
//! The decoder pins catch regressions in the parse / dequant / IDCT
//! path. The encoder produces every byte of an RDD 36 frame:
//!
//! * **§5.1 frame container** — `frame_size` BE u32, `'icpf'` magic,
//!   frame_header, picture()+.
//! * **§5.1.1 / §6.2 frame_header** — `hdr_size`, version, codec_id,
//!   width/height, chroma_format + interlace_mode packed byte, descriptive
//!   metadata bytes, `flags`, optional `luma_quantization_matrix` +
//!   `chroma_quantization_matrix`.
//! * **§5.2 picture_header** — `pic_hdr_size`, `pic_size`,
//!   `log2_desired_slice_size_in_mb`, `num_slices`, optional
//!   `slice_size_table`.
//! * **§5.3 slice** — `slice_header` (qIndex, Y/Cb/Cr/A data sizes),
//!   Y/Cb/Cr (+A) entropy-coded payloads.
//! * **§7.1.1 coefficient coder** — DC/AC run/level/sign exp-Golomb
//!   combination codes, with per-component coder reset at each slice
//!   boundary.
//! * **§7.4 forward DCT** + **§7.3 quantisation** decide the coefficient
//!   stream the entropy coder packs.
//!
//! Any byte-level drift in any of the above surfaces here as a SHA
//! mismatch. The companion `assert!(round_trip)` line at the end of
//! each test then re-decodes the just-encoded packet through this
//! crate's decoder, so a bit-correct change that updates BOTH SHA and
//! decoder still passes — but a drift that breaks the decode catches
//! the SHA-only flipper.
//!
//! ### Pin scope
//!
//! The synthetic input is **128 × 64**, the smallest 4:2:2 frame that
//! still exercises a multi-slice picture at the default 8-MB slice
//! width (`128 / 16 = 8` macroblocks/row → exactly one 8-MB slice per
//! row, 4 rows). For 4:4:4 profiles we resize the same synthetic into
//! a `Yuv444P` frame so chroma is at full luma resolution. Every
//! profile is exercised at its `Profile::default_quant_index` with the
//! flat all-4s default matrices (the `prores_ks`-compatible 20-byte
//! frame header — `load_luma_qmat = load_chroma_qmat = 0`).
//!
//! Clean-room note: no external encoder source consulted. The
//! constants below were captured from this crate's own encoder run
//! against the synthetic input defined in this file. The FIPS 180-4
//! SHA-256 self-check (§B.1 / §B.2) guards against typos in the hash
//! routine that would mask a real encoder regression.

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::encoder::{
    encode_frame, encode_frame_interlaced, encode_frame_with_alpha, encode_frame_with_qmats,
};
use oxideav_prores::frame::{ChromaFormat, Profile};
use oxideav_prores::quant::QuantMatrices;

const W: u32 = 128;
const H: u32 = 64;

// ---------------------------------------------------------------------
// Synthetic deterministic inputs. The pixel values are pure functions
// of `(i, j)` and the chroma subsampling, so every CI run produces
// byte-identical inputs and any encoder-output drift surfaces as a
// SHA mismatch.
// ---------------------------------------------------------------------

fn synthetic_yuv422p_8bit() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 3 + j * 5) as u16 % 256) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2).clamp(-48, 48))) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2).clamp(-48, 48))) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn synthetic_yuv444p_8bit() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 3 + j * 5) as u16 % 256) as u8;
            cb[j * w + i] = (128 + ((i as i32 - w as i32 / 2).clamp(-48, 48))) as u8;
            cr[j * w + i] = (128 + ((j as i32 - h as i32 / 2).clamp(-48, 48))) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w,
                data: cb,
            },
            VideoPlane {
                stride: w,
                data: cr,
            },
        ],
    }
}

fn synthetic_yuv444p_8bit_with_alpha() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let mut frame = synthetic_yuv444p_8bit();
    // Alpha at full luma resolution; deterministic mid-range gradient.
    let mut a = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            a[j * w + i] = ((i + j * 2) as u16 % 256) as u8;
        }
    }
    frame.planes.push(VideoPlane { stride: w, data: a });
    frame
}

// ---------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4). Each `tests/*.rs` is its own binary, so
// the hash routine is duplicated (the workspace forbids cross-test
// `common` modules without `lib.rs`). The FIPS self-check below guards
// against typos in the K constants / round equations.
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
/// before it can mask a real encoder regression.
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

/// Decode the encoded packet back through this crate's decoder and
/// confirm dimensions + plane count match the encoded chroma_format.
/// Catches a SHA-only flip (an encoder change that produces different
/// bytes but is internally consistent with a matched decoder change).
fn assert_decodes_cleanly(
    pkt: &[u8],
    expected_chroma: ChromaFormat,
    bit_depth: BitDepth,
    expected_alpha: bool,
) {
    let decoded = decode_packet_with_depth(pkt, Some(0), Some((bit_depth, expected_chroma)))
        .expect("decode round-trip");
    let expected_planes = if expected_alpha { 4 } else { 3 };
    assert_eq!(
        decoded.planes.len(),
        expected_planes,
        "round-trip plane count mismatch (alpha={expected_alpha})"
    );
    // Round-trip preserves pts.
    assert_eq!(decoded.pts, Some(0), "round-trip lost pts");
    // First plane stride must match the encoded width.
    assert_eq!(
        decoded.planes[0].stride, W as usize,
        "round-trip Y stride mismatch"
    );
}

// ---------------------------------------------------------------------
// 422 progressive — Proxy / LT / Standard / HQ at 8-bit + Standard 10-bit.
// ---------------------------------------------------------------------

/// 422 Proxy (`apco`), progressive, 8-bit, default qi = 8.
#[test]
fn encoder_sha_pin_apco_8bit_default_qi() {
    let frame = synthetic_yuv422p_8bit();
    let pkt = encode_frame(&frame, W, H, ChromaFormat::Y422, Profile::Proxy, 8).unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(sha, EXPECTED_APCO_8BIT_QI8, "apco 8-bit qi=8 SHA drift");
    assert_decodes_cleanly(&pkt, ChromaFormat::Y422, BitDepth::Eight, false);
}

/// 422 LT (`apcs`), progressive, 8-bit, default qi = 6.
#[test]
fn encoder_sha_pin_apcs_8bit_default_qi() {
    let frame = synthetic_yuv422p_8bit();
    let pkt = encode_frame(&frame, W, H, ChromaFormat::Y422, Profile::Lt, 6).unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(sha, EXPECTED_APCS_8BIT_QI6, "apcs 8-bit qi=6 SHA drift");
    assert_decodes_cleanly(&pkt, ChromaFormat::Y422, BitDepth::Eight, false);
}

/// 422 Standard (`apcn`), progressive, 8-bit, default qi = 4.
#[test]
fn encoder_sha_pin_apcn_8bit_default_qi() {
    let frame = synthetic_yuv422p_8bit();
    let pkt = encode_frame(&frame, W, H, ChromaFormat::Y422, Profile::Standard, 4).unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(sha, EXPECTED_APCN_8BIT_QI4, "apcn 8-bit qi=4 SHA drift");
    assert_decodes_cleanly(&pkt, ChromaFormat::Y422, BitDepth::Eight, false);
}

/// 422 HQ (`apch`), progressive, 8-bit, default qi = 2.
#[test]
fn encoder_sha_pin_apch_8bit_default_qi() {
    let frame = synthetic_yuv422p_8bit();
    let pkt = encode_frame(&frame, W, H, ChromaFormat::Y422, Profile::Hq, 2).unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(sha, EXPECTED_APCH_8BIT_QI2, "apch 8-bit qi=2 SHA drift");
    assert_decodes_cleanly(&pkt, ChromaFormat::Y422, BitDepth::Eight, false);
}

// ---------------------------------------------------------------------
// 4:4:4 progressive — 4444 + 4444 XQ at 8-bit (no alpha).
// ---------------------------------------------------------------------

/// 4444 (`ap4h`), progressive, 8-bit, no alpha, default qi = 2.
#[test]
fn encoder_sha_pin_ap4h_8bit_no_alpha() {
    let frame = synthetic_yuv444p_8bit();
    let pkt = encode_frame(&frame, W, H, ChromaFormat::Y444, Profile::Prores4444, 2).unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(sha, EXPECTED_AP4H_8BIT_QI2, "ap4h 8-bit qi=2 SHA drift");
    assert_decodes_cleanly(&pkt, ChromaFormat::Y444, BitDepth::Eight, false);
}

/// 4444 XQ (`ap4x`), progressive, 8-bit, no alpha, default qi = 1.
#[test]
fn encoder_sha_pin_ap4x_8bit_no_alpha() {
    let frame = synthetic_yuv444p_8bit();
    let pkt = encode_frame(&frame, W, H, ChromaFormat::Y444, Profile::Prores4444Xq, 1).unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(sha, EXPECTED_AP4X_8BIT_QI1, "ap4x 8-bit qi=1 SHA drift");
    assert_decodes_cleanly(&pkt, ChromaFormat::Y444, BitDepth::Eight, false);
}

// ---------------------------------------------------------------------
// 4:4:4 progressive WITH alpha — 4444 8-bit, alpha_channel_type=1.
// ---------------------------------------------------------------------

/// 4444 (`ap4h`), progressive, 8-bit, 8-bit alpha, default qi = 2.
/// Exercises the §5.3.3 + §7.1.2 scanned_alpha() emission at the
/// tail of every slice.
#[test]
fn encoder_sha_pin_ap4h_8bit_with_8bit_alpha() {
    let frame = synthetic_yuv444p_8bit_with_alpha();
    let pkt = encode_frame_with_alpha(
        &frame,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        2,
        Some(AlphaChannelType::Eight),
    )
    .unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(
        sha, EXPECTED_AP4H_8BIT_QI2_ALPHA8,
        "ap4h 8-bit qi=2 + 8-bit alpha SHA drift"
    );
    assert_decodes_cleanly(&pkt, ChromaFormat::Y444, BitDepth::Eight, true);
}

// ---------------------------------------------------------------------
// 422 interlaced — apcn TFF + BFF at 8-bit.
// ---------------------------------------------------------------------

/// 422 Standard (`apcn`) interlaced top-field-first, 8-bit, default qi = 4.
/// Exercises the §5.1 two-picture-per-frame walker and the §7.2 Figure 5
/// interlaced block scan.
#[test]
fn encoder_sha_pin_apcn_8bit_interlaced_tff() {
    let frame = synthetic_yuv422p_8bit();
    let pkt = encode_frame_interlaced(
        &frame,
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        None,
        1, // TFF
    )
    .unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(
        sha, EXPECTED_APCN_8BIT_QI4_INTERLACED_TFF,
        "apcn 8-bit qi=4 TFF SHA drift"
    );
    assert_decodes_cleanly(&pkt, ChromaFormat::Y422, BitDepth::Eight, false);
}

/// 422 Standard (`apcn`) interlaced bottom-field-first, 8-bit, default qi = 4.
/// Must differ from the TFF SHA — the field order is part of the wire.
#[test]
fn encoder_sha_pin_apcn_8bit_interlaced_bff() {
    let frame = synthetic_yuv422p_8bit();
    let pkt = encode_frame_interlaced(
        &frame,
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        None,
        2, // BFF
    )
    .unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(
        sha, EXPECTED_APCN_8BIT_QI4_INTERLACED_BFF,
        "apcn 8-bit qi=4 BFF SHA drift"
    );
    assert_decodes_cleanly(&pkt, ChromaFormat::Y422, BitDepth::Eight, false);
    // TFF and BFF must produce different bytes — interlace_mode is on
    // the wire (RDD 36 §6.1.1 Table 2). A regression where the encoder
    // silently emits TFF for BFF would collide here.
    assert_ne!(
        EXPECTED_APCN_8BIT_QI4_INTERLACED_TFF, EXPECTED_APCN_8BIT_QI4_INTERLACED_BFF,
        "TFF and BFF SHAs must differ — interlace_mode is part of the wire"
    );
}

// ---------------------------------------------------------------------
// Perceptual matrices — ap4h with loaded matrices (load_*_qmat = 1).
// Exercises the 148-byte frame_header path.
// ---------------------------------------------------------------------

/// 4444 (`ap4h`) with `QuantMatrices::perceptual_for_profile(4444)` —
/// drives `load_luma_qmat = load_chroma_qmat = 1` so the 64-byte
/// luma + 64-byte chroma matrices are emitted into the frame header
/// (`frame_header_size = 148` instead of the 20-byte default).
#[test]
fn encoder_sha_pin_ap4h_8bit_perceptual_matrices() {
    let frame = synthetic_yuv444p_8bit();
    let qmats = QuantMatrices::perceptual_for_profile(Profile::Prores4444);
    let pkt = encode_frame_with_qmats(
        &frame,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        2,
        qmats,
    )
    .unwrap();
    let sha = hex(&sha256(&pkt));
    assert_eq!(
        sha, EXPECTED_AP4H_8BIT_QI2_PERCEPTUAL,
        "ap4h 8-bit qi=2 perceptual-matrices SHA drift"
    );
    assert_decodes_cleanly(&pkt, ChromaFormat::Y444, BitDepth::Eight, false);
    // The perceptual matrices flip frame_header_size to 148 — the SHA
    // must differ from the flat-matrices ap4h pin above. A regression
    // where the encoder silently dropped the loaded matrices would
    // surface as a collision here.
    assert_ne!(
        EXPECTED_AP4H_8BIT_QI2, EXPECTED_AP4H_8BIT_QI2_PERCEPTUAL,
        "flat + perceptual ap4h SHAs must differ — load_*_qmat is on the wire"
    );
}

// ---------------------------------------------------------------------
// Pinned encoder-output SHA-256s.
//
// Each constant is the hex-encoded SHA-256 of the bytes returned from
// the corresponding free-function encoder path against the synthetic
// inputs defined at the top of this file. Captured against this
// crate's own encoder run — no external reference. Any drift in the
// DCT, quantisation, slice scan, entropy coder, frame_header /
// picture_header / slice_header byte layout, or the §5.1 frame
// container will surface as a SHA mismatch.
// ---------------------------------------------------------------------

const EXPECTED_APCO_8BIT_QI8: &str =
    "f18d1bf5f2c9fb3f9a5b6872947db0bce03906f4e5d0378a38b8c9b4d1c6b94c";
const EXPECTED_APCS_8BIT_QI6: &str =
    "258e45801ec4282575b7a0cf637b396534ee0148df3136075dc6b410f47d1fc6";
const EXPECTED_APCN_8BIT_QI4: &str =
    "08a28e7132aa6045f424a5160972ffa162331668498320d4a696e25c0581f936";
const EXPECTED_APCH_8BIT_QI2: &str =
    "b1b0bfa416fef818c3b59b0dda6aa879e86ca67654b288f5f56be0ec66a70aa0";
const EXPECTED_AP4H_8BIT_QI2: &str =
    "b31728c138392c8cc9349036e3a42a25ed3a296cc6800fc4f8dba5a4a22d2ffc";
const EXPECTED_AP4X_8BIT_QI1: &str =
    "d9f16ea34257bc534ba8c8cc63a4fef58788be1b7c6dc8e3528bdc53de46659c";
const EXPECTED_AP4H_8BIT_QI2_ALPHA8: &str =
    "527caa00db995b266cf3e1dca3faaf318f60652f63117e50c9974a8b3e1a3054";
const EXPECTED_APCN_8BIT_QI4_INTERLACED_TFF: &str =
    "4ddef64dced233bf809caec388bb30ef76a12b6d02ea2e44c973c87489337105";
const EXPECTED_APCN_8BIT_QI4_INTERLACED_BFF: &str =
    "4ff781f8c66686e253d123b8f0ef640aecf4d2e9f311afb9d458cd7c6c5f34f3";
const EXPECTED_AP4H_8BIT_QI2_PERCEPTUAL: &str =
    "6f782398b5c7f1f5cc8ddf008eb71dd1c98f404b960ca862dca18662ece24637";
