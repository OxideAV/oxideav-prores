//! Criterion benchmark for the ProRes decode hot path.
//!
//! Three representative inputs are benched: a 4:2:2 8-bit Standard
//! (`apcn`) frame, a 4:4:4 8-bit ProRes 4444 (`ap4h`) frame, and a
//! 4:2:2 10-bit frame decoded to packed-LE `Yuv422P10Le`. Every input
//! is synthesized in-process via this crate's own encoder, so the bench
//! needs no external fixtures — the encode cost is paid once during
//! setup and excluded from the measured region.
//!
//! Run with e.g.:
//! `cargo bench --bench decode -- --warm-up-time 1 --measurement-time 3`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};

use oxideav_prores::decoder::{decode_packet, decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

const W: u32 = 128;
const H: u32 = 96;

/// 8-bit 4:2:2 smooth gradient (luma full-width, chroma half-width).
fn source_422_8bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i + j) * 255 / (w + h)).min(255) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-64, 64)) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-64, 64)) as u8;
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

/// 8-bit 4:4:4 smooth gradient (all three planes full-width).
fn source_444_8bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i + j) * 255 / (w + h)).min(255) as u8;
            cb[j * w + i] = (128 + ((i as i32 - w as i32 / 2) * 2).clamp(-64, 64)) as u8;
            cr[j * w + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-64, 64)) as u8;
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

/// 10-bit 4:2:2 gradient: each sample is packed little-endian into two
/// bytes, valid range `0..=1023`.
fn source_422_10bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; cw * h * 2];
    let mut cr = vec![0u8; cw * h * 2];
    for j in 0..h {
        for i in 0..w {
            let v: u16 = ((i * 7 + j * 5) as u16 % 1024).min(1023);
            let off = (j * w + i) * 2;
            y[off] = (v & 0xFF) as u8;
            y[off + 1] = (v >> 8) as u8;
        }
        for i in 0..cw {
            let cb_v: u16 = (512 + ((i as i32 - cw as i32 / 2) * 8).clamp(-256, 256)) as u16;
            let cr_v: u16 = (512 + ((j as i32 - h as i32 / 2) * 8).clamp(-256, 256)) as u16;
            let off = (j * cw + i) * 2;
            cb[off] = (cb_v & 0xFF) as u8;
            cb[off + 1] = (cb_v >> 8) as u8;
            cr[off] = (cr_v & 0xFF) as u8;
            cr[off + 1] = (cr_v >> 8) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane {
                stride: cw * 2,
                data: cb,
            },
            VideoPlane {
                stride: cw * 2,
                data: cr,
            },
        ],
    }
}

/// Encode one frame through the registry encoder and return the packet
/// bytes. The encode cost is paid here, during bench setup, and is not
/// part of the measured decode region.
fn encode_packet(frame: &VideoFrame, format: PixelFormat) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(format);

    let mut reg = CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut encoder = reg.first_encoder(&params).expect("make_encoder");
    encoder
        .send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    let pkt = encoder.receive_packet().expect("receive_packet");
    pkt.data
}

fn bench_decode(c: &mut Criterion) {
    // 4:2:2 8-bit Standard -> apcn (pick_profile default for Yuv422P).
    let apcn = encode_packet(&source_422_8bit(W, H), PixelFormat::Yuv422P);
    // 4:4:4 8-bit -> ap4h (pick_profile default for Yuv444P).
    let ap4h = encode_packet(&source_444_8bit(W, H), PixelFormat::Yuv444P);
    // 4:2:2 10-bit, decoded to packed-LE Yuv422P10Le.
    let ten = encode_packet(&source_422_10bit(W, H), PixelFormat::Yuv422P10Le);

    let mut group = c.benchmark_group("decode_frame");

    group.bench_function("apcn_422_8bit_128x96", |b| {
        b.iter(|| decode_packet(black_box(&apcn), None).expect("decode apcn"));
    });

    group.bench_function("ap4h_444_8bit_128x96", |b| {
        b.iter(|| decode_packet(black_box(&ap4h), None).expect("decode ap4h"));
    });

    group.bench_function("apcn_422_10bit_128x96", |b| {
        b.iter(|| {
            decode_packet_with_depth(
                black_box(&ten),
                None,
                Some((BitDepth::Ten, ChromaFormat::Y422)),
            )
            .expect("decode 10-bit")
        });
    });

    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
