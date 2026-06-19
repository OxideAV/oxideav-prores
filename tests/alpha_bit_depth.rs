//! End-to-end coverage for RDD 36 §7.5.2 decoded-alpha → pixel-alpha
//! bit-depth conversion, driven entirely through this crate's own
//! encoder + decoder (no external validator).
//!
//! The decoder's output bit depth is chosen at decode time, independent
//! of the alpha channel's coded depth. A stream coded with 8-bit alpha
//! (`alpha_channel_type == 1`) therefore exercises the §7.5.2 promotion
//! arm `alphaSample = round((2^b − 1) * alpha ÷ 255)` when decoded at 10-
//! or 12-bit output, and the identity arm when decoded at 8-bit. The
//! colour-coding path uses a lossy DCT, but the alpha plane is coded
//! losslessly (§7.1.2), so the decoded alpha samples are an exact
//! function of the encoder's input alpha and the requested output depth —
//! lockable without PSNR slack.

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::encoder::encode_frame_with_alpha;
use oxideav_prores::frame::{ChromaFormat, Profile};

const W: usize = 32;
const H: usize = 32;

/// Deterministic per-pixel 8-bit alpha pattern spanning the full
/// `0..=255` domain (including both endpoints) so the §7.5.2 conversion
/// is exercised across its whole range rather than at a single opacity.
fn alpha_value(x: usize, y: usize) -> u8 {
    // A ramp that hits 0 at the origin and 255 at the far corner.
    let v = (x * 255) / (W - 1);
    let w = (y * 255) / (H - 1);
    ((v + w) / 2) as u8
}

/// Build a 4:4:4 + 8-bit-alpha source frame: flat-ish Y'CbCr (the colour
/// plane content is irrelevant to this test) plus the alpha ramp.
fn source_with_8bit_alpha() -> VideoFrame {
    let mut y = vec![0u8; W * H];
    let mut cb = vec![128u8; W * H];
    let mut cr = vec![128u8; W * H];
    let mut a = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y[j * W + i] = ((i + j) as u16 % 256) as u8;
            cb[j * W + i] = 128;
            cr[j * W + i] = 128;
            a[j * W + i] = alpha_value(i, j);
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

/// §7.5.2 reference: `alphaSample = round((2^b − 1) * alpha ÷ 255)` with
/// the §3 `round(x) = floor(x + 1/2)` convention, computed independently
/// of the decoder so the test is a genuine oracle.
fn expected_sample(alpha: u8, out: BitDepth) -> u16 {
    let max_out = out.max_value() as u64;
    let num = max_out * alpha as u64 * 2 + 255;
    (num / (255 * 2)) as u16
}

fn read_alpha_plane(frame: &VideoFrame, out: BitDepth) -> Vec<u16> {
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

fn roundtrip_at_depth(out: BitDepth) {
    let src = source_with_8bit_alpha();
    let pkt = encode_frame_with_alpha(
        &src,
        W as u32,
        H as u32,
        ChromaFormat::Y444,
        BitDepth::Eight, // colour coded at 8-bit; alpha coded 8-bit
        Profile::Prores4444,
        4, // a low quantisation index — colour fidelity irrelevant here
        Some(AlphaChannelType::Eight),
    )
    .expect("encode 4444 + 8-bit alpha");

    let frame = decode_packet_with_depth(&pkt, Some(0), Some((out, ChromaFormat::Y444)))
        .unwrap_or_else(|e| panic!("decode at {out:?} failed: {e:?}"));

    let got = read_alpha_plane(&frame, out);
    assert_eq!(
        got.len(),
        W * H,
        "alpha plane must be cropped to the {W}x{H} image"
    );
    for j in 0..H {
        for i in 0..W {
            let a = alpha_value(i, j);
            let want = expected_sample(a, out);
            assert_eq!(
                got[j * W + i],
                want,
                "§7.5.2 mismatch at ({i},{j}) out={out:?}: alpha {a} -> got {} want {want}",
                got[j * W + i]
            );
        }
    }
}

/// 8-bit alpha decoded at 8-bit output is the §7.5.2 identity arm: the
/// pixel alpha samples are the coded values themselves.
#[test]
fn alpha8_decodes_identity_at_8bit() {
    roundtrip_at_depth(BitDepth::Eight);
}

/// 8-bit alpha decoded at 10-bit output: §7.5.2 promotion by
/// `round(1023 * alpha / 255)`, end to end.
#[test]
fn alpha8_promotes_to_10bit() {
    roundtrip_at_depth(BitDepth::Ten);
}

/// 8-bit alpha decoded at 12-bit output: §7.5.2 promotion by
/// `round(4095 * alpha / 255)`, end to end.
#[test]
fn alpha8_promotes_to_12bit() {
    roundtrip_at_depth(BitDepth::Twelve);
}

/// The §7.5.2 endpoints must land exactly: a fully-transparent (0) and a
/// fully-opaque (255) source alpha map to sample 0 and `2^b − 1` at
/// every output depth.
#[test]
fn alpha8_endpoints_exact_each_depth() {
    for out in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
        assert_eq!(expected_sample(0, out), 0);
        assert_eq!(u32::from(expected_sample(255, out)), out.max_value());
    }
}

// ───────── partial bottom-MB-row alpha (RDD 36 §7.5.3) ─────────
//
// Regression lock for the per-slice scanned-alpha array length on the
// bottom macroblock row of a picture whose height is NOT a multiple of
// 16. RDD 36 §7.5.3 says the alpha array "does not include alpha values
// for the excess row(s) of pixels at the bottom of slices with
// i = height_in_mb − 1", which reads as "size the bottom-row array to the
// visible row count". Empirically that is wrong: reference ProRes 4444
// streams (and this crate, to stay bit-compatible) encode the FULL 16
// rows of the bottom MB row and the decoder discards the excess rows on
// paste. Sizing the coded array to the visible row count instead raises
// "run overruns alphaValues array" against real bitstreams. This test
// pins the working behaviour so a future "spec-literal" refactor that
// truncates the bottom-row array length is caught immediately, while also
// proving the *visible* alpha is recovered losslessly through the partial
// bottom row.

/// Width is MB-aligned (multiple of 16) but height is not: 32×24 yields a
/// 2-MB-row picture (padded height 32) whose bottom MB row carries only
/// 24 − 16 = 8 visible rows. Exercises the §7.5.3 partial-row path.
const WP: usize = 32;
const HP: usize = 24;

fn ramp_alpha(x: usize, y: usize) -> u8 {
    let v = (x * 255) / (WP - 1);
    let w = (y * 255) / (HP - 1);
    ((v + w) / 2) as u8
}

fn source_partial_height_alpha() -> VideoFrame {
    let mut y = vec![0u8; WP * HP];
    let mut cb = vec![128u8; WP * HP];
    let cr = vec![128u8; WP * HP];
    let mut a = vec![0u8; WP * HP];
    for j in 0..HP {
        for i in 0..WP {
            y[j * WP + i] = ((i + j) as u16 % 256) as u8;
            cb[j * WP + i] = 128;
            a[j * WP + i] = ramp_alpha(i, j);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: WP,
                data: y,
            },
            VideoPlane {
                stride: WP,
                data: cb,
            },
            VideoPlane {
                stride: WP,
                data: cr,
            },
            VideoPlane {
                stride: WP,
                data: a,
            },
        ],
    }
}

fn partial_roundtrip_at_depth(out: BitDepth) {
    let src = source_partial_height_alpha();
    let pkt = encode_frame_with_alpha(
        &src,
        WP as u32,
        HP as u32,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        4,
        Some(AlphaChannelType::Eight),
    )
    .expect("encode 4444 + 8-bit alpha at non-MB-aligned height");

    let frame = decode_packet_with_depth(&pkt, Some(0), Some((out, ChromaFormat::Y444)))
        .unwrap_or_else(|e| panic!("decode at {out:?} failed: {e:?}"));

    let got = read_alpha_plane(&frame, out);
    assert_eq!(
        got.len(),
        WP * HP,
        "alpha plane must crop to the visible {WP}x{HP} image (including the partial bottom MB row)"
    );
    // The whole visible image — including rows 16..24 in the partial
    // bottom MB row — must recover the §7.5.2 sample exactly (alpha is
    // lossless per §7.1.2).
    for j in 0..HP {
        for i in 0..WP {
            let a = ramp_alpha(i, j);
            let want = expected_sample(a, out);
            assert_eq!(
                got[j * WP + i],
                want,
                "§7.5.2/§7.5.3 mismatch at ({i},{j}) out={out:?} \
                 (partial bottom MB row begins at j=16): alpha {a} -> got {} want {want}",
                got[j * WP + i]
            );
        }
    }
}

/// Partial bottom-MB-row alpha roundtrips losslessly at 8-bit output.
#[test]
fn alpha8_partial_bottom_row_identity_8bit() {
    partial_roundtrip_at_depth(BitDepth::Eight);
}

/// Partial bottom-MB-row alpha roundtrips through §7.5.2 promotion at
/// 10-bit output.
#[test]
fn alpha8_partial_bottom_row_promotes_10bit() {
    partial_roundtrip_at_depth(BitDepth::Ten);
}

/// Partial bottom-MB-row alpha roundtrips through §7.5.2 promotion at
/// 12-bit output.
#[test]
fn alpha8_partial_bottom_row_promotes_12bit() {
    partial_roundtrip_at_depth(BitDepth::Twelve);
}

/// The same partial-height geometry with **16-bit** coded alpha
/// (`alpha_channel_type == 2`) — the bottom-MB-row array length and the
/// §7.5.2 16-bit demotion arm together.
#[test]
fn alpha16_partial_bottom_row_roundtrips() {
    let mut a = vec![0u8; WP * HP * 2];
    for j in 0..HP {
        for i in 0..WP {
            // Full 16-bit sweep, distinct per pixel, hitting the partial
            // bottom rows (j >= 16) with non-trivial values.
            let v = (((i + j * WP) * 65535) / (WP * HP - 1)) as u16;
            let off = (j * WP + i) * 2;
            a[off] = (v & 0xFF) as u8;
            a[off + 1] = (v >> 8) as u8;
        }
    }
    // Colour coded at 12-bit, so the Y/Cb/Cr planes must be 2 bytes per
    // sample (their content is irrelevant to the alpha lock).
    let mut y12 = vec![0u8; WP * HP * 2];
    let mut c12 = vec![0u8; WP * HP * 2];
    for px in 0..WP * HP {
        // mid-grey 12-bit: 2048 for luma, 2048 for chroma.
        y12[px * 2] = 0x00;
        y12[px * 2 + 1] = 0x08;
        c12[px * 2] = 0x00;
        c12[px * 2 + 1] = 0x08;
    }
    let src = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: WP * 2,
                data: y12,
            },
            VideoPlane {
                stride: WP * 2,
                data: c12.clone(),
            },
            VideoPlane {
                stride: WP * 2,
                data: c12,
            },
            VideoPlane {
                stride: WP * 2,
                data: a.clone(),
            },
        ],
    };
    let pkt = encode_frame_with_alpha(
        &src,
        WP as u32,
        HP as u32,
        ChromaFormat::Y444,
        BitDepth::Twelve,
        Profile::Prores4444,
        2,
        Some(AlphaChannelType::Sixteen),
    )
    .expect("encode 4444 + 16-bit alpha at non-MB-aligned height");

    let frame =
        decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Twelve, ChromaFormat::Y444)))
            .expect("decode 16-bit alpha");
    let got = read_alpha_plane(&frame, BitDepth::Twelve);
    assert_eq!(got.len(), WP * HP);
    // §7.5.2 16-bit demotion to 12-bit: round((2^12 − 1) * alpha / 65535).
    for j in 0..HP {
        for i in 0..WP {
            let off = (j * WP + i) * 2;
            let alpha = u16::from_le_bytes([a[off], a[off + 1]]) as u64;
            let want = ((4095u64 * alpha * 2 + 65535) / (65535 * 2)) as u16;
            assert_eq!(
                got[j * WP + i],
                want,
                "16-bit alpha §7.5.2 demotion mismatch at ({i},{j})"
            );
        }
    }
}
