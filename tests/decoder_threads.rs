//! Multi-threaded slice decoding (`decode_packet_with_threads`,
//! `Decoder::set_execution_context`).
//!
//! Slices are independent (RDD 36 §5.3), so the decoder may decode the
//! macroblock rows of one picture on several threads. The frame must
//! not depend on the thread count: every case here decodes the same
//! packet with 1, 2, 3, 7 and 64 threads and requires identical planes,
//! across progressive and both field orders, alpha, every output depth,
//! and a picture with partial slices at the right edge and a partial
//! macroblock row at the bottom. A truncated packet must fail with any
//! thread count.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, ExecutionContext, Frame, MediaType, Packet, PixelFormat, TimeBase,
    VideoFrame,
};
use oxideav_prores::decoder::{
    OutputRange, decode_packet_with_format, decode_packet_with_threads, make_decoder,
};
use oxideav_prores::encoder::{EncoderConfig, make_encoder_with_config};
use oxideav_prores::frame::Profile;

// 25 macroblocks across (slices of 8, 8, 8, 1) and 13 macroblock rows,
// the last one partial; odd height, so the two fields differ in size.
const W: u32 = 397;
const H: u32 = 201;

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

fn encode(pix: PixelFormat, config: EncoderConfig, alpha: bool) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let cw = if matches!(pix, PixelFormat::Yuv422P10Le) {
        w.div_ceil(2)
    } else {
        w
    };
    let mut planes = vec![
        plane(w, h, 1023, 1),
        plane(cw, h, 1023, 2),
        plane(cw, h, 1023, 3),
    ];
    if alpha {
        planes.push(plane(w, h, 65535, 4));
    }
    let mut params = CodecParameters::video(CodecId::new("prores"));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(pix);
    let mut enc = make_encoder_with_config(&params, config).expect("encoder");
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: Some(0),
        planes,
    }))
    .expect("send");
    enc.receive_packet().expect("packet").data
}

fn same(a: &VideoFrame, b: &VideoFrame) -> bool {
    a.planes.len() == b.planes.len()
        && a.planes
            .iter()
            .zip(&b.planes)
            .all(|(p, q)| p.stride == q.stride && p.data == q.data)
}

fn assert_thread_count_invariant(packet: &[u8], formats: &[PixelFormat]) {
    for &format in formats {
        let single = decode_packet_with_format(packet, None, Some(format), OutputRange::Full)
            .expect("single-threaded decode");
        for threads in [0, 1, 2, 3, 7, 64] {
            let multi =
                decode_packet_with_threads(packet, None, Some(format), OutputRange::Full, threads)
                    .expect("multi-threaded decode");
            assert!(
                same(&single, &multi),
                "{format:?}: {threads}-thread frame differs from the single-threaded one"
            );
        }
        // A packet cut inside the slice data fails for every thread
        // count rather than returning a partly decoded frame.
        let cut = &packet[..packet.len() - packet.len() / 3];
        for threads in [1, 4] {
            assert!(
                decode_packet_with_threads(cut, None, Some(format), OutputRange::Full, threads)
                    .is_err(),
                "{format:?}: truncated packet accepted with {threads} threads"
            );
        }
    }
}

#[test]
fn progressive_422_is_thread_count_invariant() {
    let packet = encode(
        PixelFormat::Yuv422P10Le,
        EncoderConfig::for_profile(Profile::Hq),
        false,
    );
    assert_thread_count_invariant(
        &packet,
        &[
            PixelFormat::Yuv422P,
            PixelFormat::Yuv422P10Le,
            PixelFormat::Yuv422P12Le,
            PixelFormat::Yuv422P16Le,
            PixelFormat::Yuva422P10Le,
        ],
    );
}

#[test]
fn interlaced_422_is_thread_count_invariant() {
    for mode in [1, 2] {
        let packet = encode(
            PixelFormat::Yuv422P10Le,
            EncoderConfig::for_profile(Profile::Standard).with_interlace_mode(mode),
            false,
        );
        assert_thread_count_invariant(&packet, &[PixelFormat::Yuv422P10Le, PixelFormat::Yuv422P]);
    }
}

#[test]
fn alpha_444_is_thread_count_invariant() {
    for mode in [0, 1, 2] {
        let packet = encode(
            PixelFormat::Yuv444P10Le,
            EncoderConfig::for_profile(Profile::Prores4444)
                .with_interlace_mode(mode)
                .with_mbs_per_slice(2),
            true,
        );
        assert_thread_count_invariant(
            &packet,
            &[
                PixelFormat::Yuv444P12Le,
                PixelFormat::Yuva444P,
                PixelFormat::Yuva444P16Le,
            ],
        );
    }
}

#[test]
fn execution_context_grants_the_thread_budget() {
    let packet = encode(
        PixelFormat::Yuv422P10Le,
        EncoderConfig::for_profile(Profile::Lt).with_interlace_mode(1),
        false,
    );
    let mut params = CodecParameters::video(CodecId::new("prores"));
    params.pixel_format = Some(PixelFormat::Yuv422P10Le);
    let decode = |ctx: Option<ExecutionContext>| {
        let mut dec = make_decoder(&params).expect("decoder");
        if let Some(ctx) = ctx {
            dec.set_execution_context(&ctx);
        }
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), packet.clone()))
            .expect("send");
        match dec.receive_frame().expect("frame") {
            Frame::Video(v) => v,
            _ => panic!("video frame expected"),
        }
    };
    let serial = decode(None);
    for threads in [1, 2, 5] {
        let granted = decode(Some(ExecutionContext::with_threads(threads)));
        assert!(same(&serial, &granted), "{threads} granted threads");
    }
}
