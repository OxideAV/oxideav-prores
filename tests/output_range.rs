//! RDD 36 §7.5.1 output-range clamp.
//!
//! §7.5.1 lets a decoder clamp reconstructed color component samples
//! either to all available quantization levels (`0 ..= 2^b − 1`,
//! [`OutputRange::Full`]) or to the permissible video levels that avoid
//! the BT.601/BT.709 synchronization/timing reference codes
//! (`1 ..= 2^b − 2`, [`OutputRange::Video`]).
//!
//! These tests encode a high-contrast frame (whose decoded samples are
//! pushed against both clamp limits), then decode it under both ranges
//! and assert:
//!   1. The `Video` color planes never contain the extreme codes `0`
//!      or `2^b − 1`.
//!   2. Every `Video` color sample equals its `Full` counterpart except
//!      where `Full` clamped to an extreme, in which case `Video` is
//!      exactly one level inside (`0 → 1`, `max → max − 1`).
//!   3. The default (`Full`) decode is byte-identical to the legacy
//!      `decode_packet_with_depth` path, so the new option is additive.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame};
use oxideav_prores::decoder::{
    decode_packet_with_depth, decode_packet_with_options, BitDepth, OutputRange,
};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::ChromaFormat;

/// High-contrast 4:2:2 source for bit depth `bd`: left half luma at the
/// floor (0, forces the clamp floor on decode), right half at the
/// ceiling (`2^b − 1`, forces the ceiling), full chroma swing. Samples
/// are laid out to match the byte width the encoder expects for `pf`.
fn contrast_422(
    width: u32,
    height: u32,
    bd: BitDepth,
    pf: PixelFormat,
) -> (VideoFrame, CodecParameters) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let bps = bd.bytes_per_sample();
    let hi = bd.max_value() as u16;

    let put = |buf: &mut [u8], idx: usize, val: u16| match bps {
        1 => buf[idx] = val as u8,
        _ => {
            buf[idx * 2] = (val & 0xFF) as u8;
            buf[idx * 2 + 1] = (val >> 8) as u8;
        }
    };

    let mut y = vec![0u8; w * h * bps];
    let mut cb = vec![0u8; cw * h * bps];
    let mut cr = vec![0u8; cw * h * bps];
    for j in 0..h {
        for i in 0..w {
            put(&mut y, j * w + i, if i < w / 2 { 0 } else { hi });
        }
        for i in 0..cw {
            put(&mut cb, j * cw + i, if i < cw / 2 { 0 } else { hi });
            put(&mut cr, j * cw + i, if j < h / 2 { 0 } else { hi });
        }
    }
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * bps,
                data: y,
            },
            VideoPlane {
                stride: cw * bps,
                data: cb,
            },
            VideoPlane {
                stride: cw * bps,
                data: cr,
            },
        ],
    };
    let mut params = CodecParameters::video(CodecId::new("prores"));
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pf);
    (frame, params)
}

fn encode_one(frame: &VideoFrame, params: &CodecParameters) -> Vec<u8> {
    let cfg = EncoderConfig::default();
    let mut enc = make_encoder_with_config(params, cfg).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    enc.receive_packet().expect("receive_packet").data
}

/// Read color sample `(plane, idx)` honouring the requested bit depth.
fn sample(data: &[u8], idx: usize, bd: BitDepth) -> u32 {
    match bd {
        BitDepth::Eight => data[idx] as u32,
        BitDepth::Ten | BitDepth::Twelve => {
            u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]) as u32
        }
    }
}

fn check_range(bd: BitDepth, pf: PixelFormat) {
    let (frame, params) = contrast_422(64, 48, bd, pf);
    let packet = encode_one(&frame, &params);

    let requested = Some((bd, ChromaFormat::Y422));
    let full = decode_packet_with_options(&packet, None, requested, OutputRange::Full)
        .expect("full decode");
    let video = decode_packet_with_options(&packet, None, requested, OutputRange::Video)
        .expect("video decode");

    let max = bd.max_value();
    assert_eq!(full.planes.len(), video.planes.len());

    let mut full_hit_floor = 0usize;
    let mut full_hit_ceil = 0usize;

    // Compare the three color planes (Y, Cb, Cr).
    for p in 0..3 {
        let fd = &full.planes[p].data;
        let vd = &video.planes[p].data;
        assert_eq!(fd.len(), vd.len(), "plane {p} length");
        let count = fd.len() / bd.bytes_per_sample();
        for i in 0..count {
            let f = sample(fd, i, bd);
            let v = sample(vd, i, bd);
            // (1) Video output never uses the reserved extreme codes.
            assert!(
                v >= 1 && v < max,
                "plane {p} sample {i}: video value {v} outside 1..={}",
                max - 1
            );
            // (2) Video tracks Full except at clamped extremes.
            match f {
                0 => {
                    assert_eq!(v, 1, "plane {p} sample {i}: floor must map 0->1");
                    full_hit_floor += 1;
                }
                x if x == max => {
                    assert_eq!(v, max - 1, "plane {p} sample {i}: ceil must map max->max-1");
                    full_hit_ceil += 1;
                }
                _ => assert_eq!(v, f, "plane {p} sample {i}: interior values must match"),
            }
        }
    }

    // The high-contrast source must actually exercise both clamp limits,
    // otherwise the test would pass vacuously.
    assert!(full_hit_floor > 0, "{bd:?}: source never reached the floor");
    assert!(
        full_hit_ceil > 0,
        "{bd:?}: source never reached the ceiling"
    );

    // (3) Default decode == legacy path == Full.
    let legacy = decode_packet_with_depth(&packet, None, requested).expect("legacy decode");
    for p in 0..full.planes.len() {
        assert_eq!(
            legacy.planes[p].data, full.planes[p].data,
            "plane {p}: decode_packet_with_depth must equal OutputRange::Full"
        );
    }
}

#[test]
fn output_range_8bit() {
    check_range(BitDepth::Eight, PixelFormat::Yuv422P);
}

#[test]
fn output_range_10bit() {
    check_range(BitDepth::Ten, PixelFormat::Yuv422P10Le);
}

#[test]
fn output_range_12bit() {
    check_range(BitDepth::Twelve, PixelFormat::Yuv422P12Le);
}
