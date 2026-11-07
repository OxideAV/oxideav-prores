//! Multi-threaded slice coding (`EncoderConfig::threads`).
//!
//! Slices are independent (RDD 36 §5.3), so the encoder may code the
//! slices of one picture on several threads and join them in order. The
//! coded frame must not depend on the thread count or on scheduling:
//! every case here encodes the same frame with 1, 2, 3, 7 and 64
//! threads and requires byte-identical packets, across progressive,
//! interlaced, alpha and rate-controlled encodes on a picture with
//! partial slices at the right edge and a partial macroblock row at the
//! bottom.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::decoder::decode_packet;
use oxideav_prores::encoder::{EncoderConfig, make_encoder_with_config};
use oxideav_prores::frame::Profile;

// 25 macroblocks across (slices of 8, 8, 8, 1) and 13 macroblock rows,
// the last one partial: 52 slices per progressive picture.
const W: u32 = 400;
const H: u32 = 200;

fn params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("prores"));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

/// Little-endian 16-bit plane of noisy texture, so slices differ in
/// coded size and finish at different times.
fn plane(w: usize, h: usize, max: u32, seed: u32) -> VideoPlane {
    let mut s = seed;
    let mut data = Vec::with_capacity(w * h * 2);
    for j in 0..h {
        for i in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((i * 7 + j * 3) as u32 + (s >> 24)) % (max + 1);
            data.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
    VideoPlane {
        stride: w * 2,
        data,
    }
}

fn frame(chroma_w: usize, alpha: bool) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let mut planes = vec![
        plane(w, h, 1023, 1),
        plane(chroma_w, h, 1023, 2),
        plane(chroma_w, h, 1023, 3),
    ];
    if alpha {
        planes.push(plane(w, h, 65535, 4));
    }
    VideoFrame {
        pts: Some(0),
        planes,
    }
}

fn encode(params: &CodecParameters, config: EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let mut enc = make_encoder_with_config(params, config).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    enc.receive_packet().expect("packet").data
}

fn assert_thread_count_invariant(
    params: CodecParameters,
    config: EncoderConfig,
    frame: VideoFrame,
) -> Vec<u8> {
    let single = encode(&params, config.clone(), &frame);
    decode_packet(&single, None).expect("single-threaded packet decodes");
    for threads in [2, 3, 7, 64] {
        let multi = encode(&params, config.clone().with_threads(threads), &frame);
        assert!(
            multi == single,
            "{threads}-thread packet differs from the single-threaded one"
        );
    }
    single
}

#[test]
fn progressive_422_is_thread_count_invariant() {
    assert_thread_count_invariant(
        params(PixelFormat::Yuv422P10Le),
        EncoderConfig::signature_for_profile(Profile::Hq),
        frame(W as usize / 2, false),
    );
}

#[test]
fn interlaced_422_is_thread_count_invariant() {
    assert_thread_count_invariant(
        params(PixelFormat::Yuv422P10Le),
        EncoderConfig::signature_for_profile(Profile::Standard).with_interlace_mode(1),
        frame(W as usize / 2, false),
    );
}

#[test]
fn alpha_444_is_thread_count_invariant() {
    assert_thread_count_invariant(
        params(PixelFormat::Yuv444P10Le),
        EncoderConfig::signature_for_profile(Profile::Prores4444),
        frame(W as usize, true),
    );
}

#[test]
fn rate_controlled_422_is_thread_count_invariant() {
    // A target well below the qi-4 size forces several trial encodes.
    let mut p = params(PixelFormat::Yuv422P10Le);
    p.frame_rate = Some(oxideav_core::Rational::new(25, 1));
    p.bit_rate = Some(25 * 8 * 20_000);
    let config = EncoderConfig::for_profile(Profile::Standard);
    let unconstrained = encode(&p, config.clone(), &frame(W as usize / 2, false));
    assert!(
        unconstrained.len() > 40_000,
        "target must force rate control"
    );
    let coded =
        assert_thread_count_invariant(p, config.with_rate_control(), frame(W as usize / 2, false));
    assert!(coded.len() < unconstrained.len());
}

#[test]
fn small_slice_counts_and_all_threads() {
    // One macroblock row of one slice: more threads than work.
    let mut p = params(PixelFormat::Yuv422P10Le);
    p.width = Some(16);
    p.height = Some(16);
    let f = VideoFrame {
        pts: Some(0),
        planes: vec![
            plane(16, 16, 1023, 5),
            plane(8, 16, 1023, 6),
            plane(8, 16, 1023, 7),
        ],
    };
    let run = |config: EncoderConfig| {
        let mut enc = make_encoder_with_config(&p, config).expect("encoder");
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        enc.receive_packet().expect("packet").data
    };
    let config = EncoderConfig::for_profile(Profile::Hq);
    assert_eq!(run(config.clone()), run(config.clone().with_threads(8)));
    assert_eq!(run(config.clone()), run(config.with_all_threads()));
}
