//! `EncoderConfig::meta` + automatic `frame_rate_code` derivation.
//!
//! RDD 36 §6.2 / Table 4 names a handful of `frame_rate_code` values
//! (24/1.001, 24, 25, 30/1.001, 30, 50, 60/1.001, 60, 100, 120/1.001,
//! 120). The encoder writes whatever code is selected into byte 13 of
//! the frame header alongside `aspect_ratio_information`. This driver
//! checks two things:
//!
//! 1. When `EncoderConfig::meta` is left at `None`, the encoder derives
//!    `frame_rate_code` from `CodecParameters::frame_rate` via
//!    [`oxideav_prores::frame::frame_rate_code_from_rational`] and
//!    leaves the rest at 0.
//! 2. When `EncoderConfig::meta` is `Some(meta)`, every byte is
//!    written verbatim and overrides the derived rate.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, VideoFrame};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{
    alpha_channel_type_from_code, aspect_ratio_from_code, color_primaries_from_code,
    matrix_coefficients_from_code, parse_frame, rational_from_frame_rate_code, AlphaChannelType,
    ColorPrimaries, FrameMeta, MatrixCoefficients,
};
use oxideav_prores::CODEC_ID_STR;

fn synth_422(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i + j) * 200 / (w + h)).min(255) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-50, 50)) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-50, 50)) as u8;
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

fn make_params(width: u32, height: u32, frame_rate: Option<Rational>) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.media_type = MediaType::Video;
    p.width = Some(width);
    p.height = Some(height);
    p.pixel_format = Some(PixelFormat::Yuv422P);
    p.frame_rate = frame_rate;
    p
}

fn encode_with(width: u32, height: u32, fr: Option<Rational>, cfg: EncoderConfig) -> Vec<u8> {
    let params = make_params(width, height, fr);
    let src = synth_422(width, height);
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(src)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    pkt.data
}

#[test]
fn frame_rate_code_derived_from_codec_params() {
    // params.frame_rate == 30000/1001 (= NTSC drop-frame). The encoder
    // must populate frame_rate_code = 4 in the header even though the
    // EncoderConfig has no explicit meta.
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(30_000, 1001)),
        EncoderConfig::default(),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(fh.frame_rate_code, 4, "30000/1001 → code 4");
    // Other meta fields stay at 0 (the derivation only touches frame_rate_code).
    assert_eq!(fh.aspect_ratio_information, 0);
    assert_eq!(fh.color_primaries, 0);
    assert_eq!(fh.transfer_characteristic, 0);
    assert_eq!(fh.matrix_coefficients, 0);
}

#[test]
fn frame_rate_code_zero_for_unknown_rate() {
    // 90 fps is not in Table 4 — encoder writes code 0.
    let pkt = encode_with(64, 48, Some(Rational::new(90, 1)), EncoderConfig::default());
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(fh.frame_rate_code, 0);
}

#[test]
fn frame_rate_code_zero_when_params_missing_rate() {
    let pkt = encode_with(64, 48, None, EncoderConfig::default());
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(fh.frame_rate_code, 0);
    assert_eq!(fh.aspect_ratio_information, 0);
}

#[test]
fn explicit_meta_overrides_derivation() {
    // params says 25 fps (would derive code 3) but explicit meta pins
    // code 8 (60 fps) + 16:9 aspect + BT.2020 — and every field shows
    // up in the parsed header.
    let meta = FrameMeta {
        aspect_ratio_information: 3,
        frame_rate_code: 8,
        color_primaries: 9,
        transfer_characteristic: 16,
        matrix_coefficients: 9,
    };
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(25, 1)),
        EncoderConfig::default().with_meta(meta),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(fh.frame_rate_code, 8);
    assert_eq!(fh.aspect_ratio_information, 3);
    assert_eq!(fh.color_primaries, 9);
    assert_eq!(fh.transfer_characteristic, 16);
    assert_eq!(fh.matrix_coefficients, 9);
}

#[test]
fn explicit_unknown_meta_keeps_zero_when_params_have_rate() {
    // params has 30 fps but explicit meta is FrameMeta::unknown() —
    // the explicit override wins, so frame_rate_code stays 0 even
    // though the rate was known.
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(30, 1)),
        EncoderConfig::default().with_meta(FrameMeta::unknown()),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(fh.frame_rate_code, 0);
}

#[test]
fn frame_rate_code_default_path_byte_compatible_with_legacy() {
    // No frame_rate set + no explicit meta = every byte 0 — must match
    // the byte-for-byte behaviour of the pre-FrameMeta encoder.
    let pkt = encode_with(64, 48, None, EncoderConfig::default());
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    // Byte 13 (aspect_ratio_information(4) + frame_rate_code(4)) = 0.
    assert_eq!(pkt[8 + 2 + 1 + 1 + 4 + 2 + 2 + 1 + 1], 0);
    assert_eq!(fh.aspect_ratio_information, 0);
    assert_eq!(fh.frame_rate_code, 0);
}

/// End-to-end: encode through the high-level [`Encoder`] trait with
/// `CodecParameters::frame_rate` set to a real-world NTSC drop-frame
/// rate, parse the emitted bytes back through [`parse_frame`], and
/// recover the original [`oxideav_core::Rational`] via
/// [`rational_from_frame_rate_code`]. This is the canonical downstream
/// consumer path: a demuxer hands a ProRes packet to the decoder, and
/// the pipeline wants to surface the source's frame rate without
/// having to know RDD 36 §6.2 / Table 4 itself. The forward derivation
/// already lives in `encode_with`; the reverse closure is the new
/// surface added this round.
#[test]
fn encoder_derived_rate_round_trips_through_decoder_helper() {
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(30_000, 1001)),
        EncoderConfig::default(),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(fh.frame_rate_code, 4);
    assert_eq!(
        rational_from_frame_rate_code(fh.frame_rate_code),
        Some(Rational::new(30_000, 1001)),
        "code 4 must reverse to the exact 30000/1001 NTSC fraction",
    );
    // aspect_ratio_information was left at 0 (no explicit meta) — that's
    // "unknown", distinct from "1:1 square" or any reserved code.
    assert_eq!(aspect_ratio_from_code(fh.aspect_ratio_information), None);
}

/// Explicit-meta path: an encoder asked for 16:9 + 60 fps must surface
/// those exact codes through the decoder helpers, even though
/// `CodecParameters::frame_rate` says something else.
#[test]
fn encoder_explicit_meta_round_trips_through_decoder_helpers() {
    let meta = FrameMeta {
        aspect_ratio_information: 3, // 16:9 per Table 3
        frame_rate_code: 8,          // 60 fps per Table 4
        color_primaries: 9,
        transfer_characteristic: 16,
        matrix_coefficients: 9,
    };
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(25, 1)), // would derive code 3 if no explicit
        EncoderConfig::default().with_meta(meta),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(
        rational_from_frame_rate_code(fh.frame_rate_code),
        Some(Rational::new(60, 1)),
    );
    assert_eq!(
        aspect_ratio_from_code(fh.aspect_ratio_information),
        Some(Rational::new(16, 9)),
    );
}

/// Symmetric anti-coverage: a packet emitted with every meta field
/// at 0 must surface `None` through both helpers — the Option
/// discriminant is the bit-for-bit distinction between an unknown
/// rate and a rate of "code 0" (which doesn't exist). A regression
/// where `rational_from_frame_rate_code(0)` silently returned
/// `Some(Rational::new(0, 1))` would flip this assertion red.
#[test]
fn encoder_default_meta_round_trips_as_none_through_decoder_helpers() {
    let pkt = encode_with(64, 48, None, EncoderConfig::default());
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(rational_from_frame_rate_code(fh.frame_rate_code), None);
    assert_eq!(aspect_ratio_from_code(fh.aspect_ratio_information), None);
    // Color metadata defaults also reverse to the appropriate
    // discriminants: `color_primaries`/`matrix_coefficients` carry "0
    // = unknown" → None, and `alpha_channel_type` carries the named
    // "0 = no alpha" → Some(None) (the variant, not the outer wrap).
    assert_eq!(color_primaries_from_code(fh.color_primaries), None);
    assert_eq!(matrix_coefficients_from_code(fh.matrix_coefficients), None);
    assert_eq!(
        alpha_channel_type_from_code(fh.alpha_channel_type),
        Some(AlphaChannelType::None),
    );
}

/// End-to-end through the high-level `Encoder` trait, mirroring
/// `encoder_explicit_meta_round_trips_through_decoder_helpers` (which
/// covers Tables 3 + 4) but exercising the §6.1.1 Tables 5, 6, 7
/// reverse helpers. The encoder is asked for BT.709 primaries +
/// BT.709 matrix; we parse the emitted bytes back and convert the
/// parsed u8 codes into named [`ColorPrimaries`] /
/// [`MatrixCoefficients`] variants. This is the canonical
/// downstream-pipeline use: a colour-management stage reading a
/// decoded packet wants the source's gamut + Y'CbCr matrix without
/// reproducing Tables 5 + 6 itself.
#[test]
fn encoder_color_metadata_round_trips_through_decoder_helpers_bt709() {
    let meta = FrameMeta {
        aspect_ratio_information: 3,
        frame_rate_code: 4,
        color_primaries: 1, // BT.709
        transfer_characteristic: 1,
        matrix_coefficients: 1, // BT.709
    };
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(30_000, 1001)),
        EncoderConfig::default().with_meta(meta),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(
        color_primaries_from_code(fh.color_primaries),
        Some(ColorPrimaries::Bt709),
    );
    assert_eq!(
        matrix_coefficients_from_code(fh.matrix_coefficients),
        Some(MatrixCoefficients::Bt709),
    );
    // BT.709 K_R / K_G / K_B straight off the named variant, no
    // intermediate Table 6 lookup at the call site.
    let (k_r, k_g, k_b) = MatrixCoefficients::Bt709.luma_coefficients();
    assert_eq!((k_r, k_g, k_b), (0.2126, 0.7152, 0.0722));
}

/// BT.2020 / SMPTE ST 2084 (PQ HDR) profile exercising every named
/// helper at once — including the BT.2020 NCL matrix's distinct
/// K-triple. The transfer_characteristic byte (16 = PQ) is checked
/// as a raw byte: §6.1.1 names the formulae for codes 1 / 16 / 18
/// inline but doesn't supply a Table that this round's helper sweep
/// covers, so the field's reverse mapping stays a caller concern.
#[test]
fn encoder_color_metadata_round_trips_through_decoder_helpers_bt2020_pq() {
    let meta = FrameMeta {
        aspect_ratio_information: 3,
        frame_rate_code: 8,
        color_primaries: 9,          // BT.2020
        transfer_characteristic: 16, // PQ
        matrix_coefficients: 9,      // BT.2020 NCL
    };
    let pkt = encode_with(
        64,
        48,
        Some(Rational::new(60, 1)),
        EncoderConfig::default().with_meta(meta),
    );
    let (fh, _) = parse_frame(&pkt).expect("parse_frame");
    assert_eq!(
        color_primaries_from_code(fh.color_primaries),
        Some(ColorPrimaries::Bt2020),
    );
    assert_eq!(
        matrix_coefficients_from_code(fh.matrix_coefficients),
        Some(MatrixCoefficients::Bt2020Ncl),
    );
    assert_eq!(fh.transfer_characteristic, 16);
    let (k_r, k_g, k_b) = MatrixCoefficients::Bt2020Ncl.luma_coefficients();
    assert_eq!((k_r, k_g, k_b), (0.2627, 0.6780, 0.0593));
}

/// The transcode-forwarding chain `parse_frame` → [`FrameHeader::meta`]
/// → [`EncoderConfig::with_meta`] preserves every descriptive RDD 36
/// §5.1.1 / §6.2 metadata byte across a full decode-side parse +
/// re-encode, end-to-end through the high-level `Encoder` trait.
/// Stream A pins a BT.2020 / ST 2084 PQ HDR profile (every field a
/// named code); stream B is encoded from `fh_a.meta()` with
/// *different* `CodecParameters::frame_rate` (25 fps would derive
/// Table 4 code 3) — proving the forwarded meta overrides the
/// derivation instead of being silently recomputed, exactly as a
/// transcode pipeline needs.
///
/// [`FrameHeader::meta`]: oxideav_prores::frame::FrameHeader::meta
#[test]
fn parsed_header_meta_forwards_into_reencode_via_with_meta() {
    let src = FrameMeta {
        aspect_ratio_information: 3, // 16:9 (Table 3)
        frame_rate_code: 8,          // 60 fps (Table 4)
        color_primaries: 9,          // BT.2020 (Table 5)
        transfer_characteristic: 16, // SMPTE ST 2084 PQ (§6.1.1)
        matrix_coefficients: 9,      // BT.2020 NCL (Table 6)
    };
    let pkt_a = encode_with(
        64,
        48,
        Some(Rational::new(60, 1)),
        EncoderConfig::default().with_meta(src),
    );
    let (fh_a, _) = parse_frame(&pkt_a).expect("parse_frame A");
    assert_eq!(fh_a.meta(), src, "stream A must carry the source meta");

    // Re-encode forwarding the parsed header's folded meta. The 25 fps
    // params rate would derive frame_rate_code 3 if the meta were
    // dropped; the forwarded code 8 must win.
    let pkt_b = encode_with(
        64,
        48,
        Some(Rational::new(25, 1)),
        EncoderConfig::default().with_meta(fh_a.meta()),
    );
    let (fh_b, _) = parse_frame(&pkt_b).expect("parse_frame B");
    assert_eq!(
        fh_b.meta(),
        fh_a.meta(),
        "re-encoded stream must carry the forwarded meta verbatim"
    );
    assert_eq!(fh_b.frame_rate_code, 8, "forwarded code beats derivation");

    // Anti-coverage: the same re-encode WITHOUT forwarding derives
    // code 3 from the 25 fps params and zeroes the colour fields —
    // confirming the forwarding (not coincidence) preserved the meta.
    let pkt_c = encode_with(64, 48, Some(Rational::new(25, 1)), EncoderConfig::default());
    let (fh_c, _) = parse_frame(&pkt_c).expect("parse_frame C");
    assert_eq!(fh_c.frame_rate_code, 3);
    assert_eq!(fh_c.color_primaries, 0);
    assert_ne!(fh_c.meta(), fh_b.meta());
}
