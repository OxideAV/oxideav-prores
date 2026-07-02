//! RDD 36 §6.1.1 quantization-matrix carriage — inverse property.
//!
//! `QuantMatrices::wire_flags` picks the smallest header that reproduces a
//! matrix pair; the decoder's §6.1.1 fallback reconstructs the pair from
//! that header. Together they must be exact inverses: for *any* valid
//! `(luma, chroma)` pair, encoding then parsing the frame header yields
//! `luma_qmat == luma` and `chroma_qmat == chroma`, and the frame-header
//! size equals `20 + 64·load_luma + 64·load_chroma`.
//!
//! This exercises a broad pseudo-random spread of pairs plus the
//! structural edge relationships (equal, one-default, both-default),
//! generalising the hand-picked cases in `quant_matrix_carriage.rs` /
//! `quant_matrix_fallback.rs`. Validator-independent.

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::encode_frame_with_qmats;
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile};
use oxideav_prores::quant::{QuantMatrices, DEFAULT_QMAT};

const W: u32 = 32;
const H: u32 = 16;

fn synth() -> VideoFrame {
    let (w, h, cw) = (W as usize, H as usize, W as usize / 2);
    let mk = |n: usize, seed: usize| {
        (0..n)
            .map(|i| (((i * 37 + seed) & 0xFF) as u8) ^ ((i >> 3) as u8))
            .collect::<Vec<u8>>()
    };
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w,
                data: mk(w * h, 3),
            },
            VideoPlane {
                stride: cw,
                data: mk(cw * h, 90),
            },
            VideoPlane {
                stride: cw,
                data: mk(cw * h, 150),
            },
        ],
    }
}

/// Fill a matrix with pseudo-random valid weights in `2..=63`.
fn rand_matrix(rng: &mut u64) -> [u8; 64] {
    let mut m = [0u8; 64];
    for w in m.iter_mut() {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *w = 2 + ((*rng >> 33) % 62) as u8; // 2..=63
    }
    m
}

fn check(luma: [u8; 64], chroma: [u8; 64]) {
    let frame = synth();
    let qm = QuantMatrices { luma, chroma };
    let (want_ll, want_lc) = qm.wire_flags();
    let pkt = encode_frame_with_qmats(
        &frame,
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        qm,
    )
    .expect("encode");
    let (fh, _) = parse_frame(&pkt).expect("parse");

    // Wire flags match wire_flags()'s minimal-carriage decision.
    assert_eq!(fh.load_luma_quantization_matrix, want_ll);
    assert_eq!(fh.load_chroma_quantization_matrix, want_lc);

    // Reconstruction is the exact inverse of the carriage.
    assert_eq!(fh.luma_qmat, luma, "luma reconstruction");
    assert_eq!(fh.chroma_qmat, chroma, "chroma reconstruction");

    // Header size follows the carried table count.
    let expect = 20 + if want_ll { 64 } else { 0 } + if want_lc { 64 } else { 0 };
    assert_eq!(fh.frame_header_size as usize, expect);
}

#[test]
fn wire_flags_and_fallback_are_inverses_random() {
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..96 {
        let luma = rand_matrix(&mut rng);
        let chroma = rand_matrix(&mut rng);
        check(luma, chroma);
    }
}

#[test]
fn wire_flags_and_fallback_are_inverses_structural() {
    let mut rng: u64 = 0xD1B54A32D192ED03;
    // both default
    check(DEFAULT_QMAT, DEFAULT_QMAT);
    for _ in 0..24 {
        let a = rand_matrix(&mut rng);
        let b = rand_matrix(&mut rng);
        // luma == chroma (custom, chroma copies luma → flags 1,0)
        check(a, a);
        // default luma, custom chroma → flags 0,1
        check(DEFAULT_QMAT, a);
        // custom luma, default chroma → chroma differs from luma → flags 1,1
        check(a, DEFAULT_QMAT);
        // both custom, distinct → flags 1,1
        if a != b {
            check(a, b);
        }
    }
}

/// A single entry differing from the default flips the corresponding load
/// flag; the reconstruction still matches exactly.
#[test]
fn single_weight_perturbation() {
    for k in 0..64 {
        let mut luma = DEFAULT_QMAT;
        luma[k] = if DEFAULT_QMAT[k] == 63 { 62 } else { 63 };
        check(luma, DEFAULT_QMAT); // luma custom, chroma default → (1,1)
        let mut chroma = DEFAULT_QMAT;
        chroma[k] = 40;
        check(DEFAULT_QMAT, chroma); // default luma, custom chroma → (0,1)
    }
}
