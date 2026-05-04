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
use oxideav_prores::frame::{parse_frame, FrameMeta};
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
