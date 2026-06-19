//! Interlaced 4444 + alpha at a NON-macroblock-aligned field height —
//! validator-independent (this crate's own encoder + decoder).
//!
//! Broadcast interlaced ProRes routinely produces field heights that are
//! not multiples of 16 (e.g. 1080i → 540-row fields, 540 mod 16 = 12).
//! This combines three decode features the milestone cares about:
//!
//!   * the RDD 36 §6.2 field split (`top = (h+1)/2`, `bottom = h/2`) into
//!     two `picture()` structures,
//!   * the §7.5.3 deinterleave (top field → frame rows {0,2,4,…}, bottom
//!     field → rows {1,3,5,…}),
//!   * the §7.5.3 per-slice scanned-alpha array on the *partial bottom MB
//!     row* of each field picture (the array carries a full 16 rows; the
//!     decoder discards the excess — see `alpha_bit_depth.rs` for the
//!     reference-bitstream analysis).
//!
//! Alpha is coded losslessly (§7.1.2), so the decoded alpha plane is an
//! exact function of the source alpha and the §7.5.2 conversion — lockable
//! without PSNR slack even though the colour path is lossy.

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::encoder::encode_frame_interlaced;
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile};

// 32 wide (MB-aligned), 36 tall (NOT a multiple of 16, and each field is
// 18 rows tall → 2 MB rows with a 2-row partial bottom MB row).
const W: usize = 32;
const H: usize = 36;

/// Distinct-per-pixel 8-bit alpha so a swapped-field or off-by-one row
/// bug shows up as a mismatch. Even frame rows are bright, odd rows dim,
/// so a TFF/BFF swap inverts the recovered pattern.
fn alpha_at(x: usize, y: usize) -> u8 {
    let base = ((x * 7 + y * 13) % 200) as u8;
    if y % 2 == 0 {
        base.saturating_add(40)
    } else {
        base / 2
    }
}

fn source_interlaced_alpha() -> VideoFrame {
    let mut y = vec![0u8; W * H];
    let cb = vec![128u8; W * H];
    let cr = vec![128u8; W * H];
    let mut a = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y[j * W + i] = ((i * 3 + j * 5) % 256) as u8;
            a[j * W + i] = alpha_at(i, j);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: W, data: y },
            VideoPlane {
                stride: W,
                data: cb,
            },
            VideoPlane {
                stride: W,
                data: cr,
            },
            VideoPlane { stride: W, data: a },
        ],
    }
}

/// §7.5.2 8-bit-alpha conversion `round((2^b − 1) * alpha ÷ 255)`,
/// computed independently as the oracle.
fn expected_sample(alpha: u8, out: BitDepth) -> u16 {
    let max_out = out.max_value() as u64;
    ((max_out * alpha as u64 * 2 + 255) / (255 * 2)) as u16
}

fn read_alpha(frame: &VideoFrame, out: BitDepth) -> Vec<u16> {
    assert_eq!(frame.planes.len(), 4, "expected Y/Cb/Cr/A planes");
    let plane = &frame.planes[3];
    match out {
        BitDepth::Eight => plane.data.iter().map(|&b| b as u16).collect(),
        BitDepth::Ten | BitDepth::Twelve => plane
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
    }
}

fn roundtrip(interlace_mode: u8, out: BitDepth) {
    let src = source_interlaced_alpha();
    let pkt = encode_frame_interlaced(
        &src,
        W as u32,
        H as u32,
        ChromaFormat::Y444,
        BitDepth::Eight, // colour coded 8-bit; alpha coded 8-bit
        Profile::Prores4444,
        4,
        Some(AlphaChannelType::Eight),
        interlace_mode,
    )
    .expect("encode interlaced 4444 + alpha at non-MB-aligned field height");

    // The header must report two pictures and the requested field order.
    let (fh, _) = parse_frame(&pkt).expect("parse our packet");
    assert_eq!(fh.interlace_mode, interlace_mode, "interlace_mode");
    assert_eq!(fh.picture_count(), 2, "interlaced frame carries 2 pictures");
    assert_eq!(
        fh.alpha_channel_type, 1,
        "8-bit alpha → alpha_channel_type=1"
    );

    let frame = decode_packet_with_depth(&pkt, Some(0), Some((out, ChromaFormat::Y444)))
        .unwrap_or_else(|e| panic!("decode im={interlace_mode} out={out:?} failed: {e:?}"));

    let got = read_alpha(&frame, out);
    assert_eq!(
        got.len(),
        W * H,
        "alpha plane crops to the visible {W}x{H} frame (both fields, incl. partial bottom MB rows)"
    );
    // Every visible pixel of the reinterleaved frame — including the
    // partial bottom MB row of each field — must recover the §7.5.2
    // sample exactly.
    for j in 0..H {
        for i in 0..W {
            let want = expected_sample(alpha_at(i, j), out);
            assert_eq!(
                got[j * W + i],
                want,
                "interlaced alpha mismatch at ({i},{j}) im={interlace_mode} out={out:?}"
            );
        }
    }
}

#[test]
fn interlaced_tff_alpha_partial_field_8bit() {
    roundtrip(1, BitDepth::Eight);
}

#[test]
fn interlaced_bff_alpha_partial_field_8bit() {
    roundtrip(2, BitDepth::Eight);
}

#[test]
fn interlaced_tff_alpha_partial_field_promotes_12bit() {
    roundtrip(1, BitDepth::Twelve);
}

#[test]
fn interlaced_bff_alpha_partial_field_promotes_10bit() {
    roundtrip(2, BitDepth::Ten);
}
