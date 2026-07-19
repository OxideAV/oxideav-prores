//! Criterion benchmark for the ProRes encode hot path.
//!
//! Drives the public standalone encoder across all six RDD 36 profiles
//! on a synthetic gradient frame: the four 4:2:2 profiles (Proxy / LT /
//! Standard / HQ, fourccs `apco`/`apcs`/`apcn`/`apch`) on 8-bit 4:2:2
//! input, and the two 4:4:4 profiles (4444 / 4444 XQ, `ap4h`/`ap4x`) on
//! 8-bit 4:4:4 input. Each profile is benched at its default
//! `quantization_index`. Three extra cases exercise the deeper-bit,
//! interlaced, and alpha paths: a 10-bit 4:2:2 Standard encode, an
//! interlaced (top-field-first) 8-bit 4:2:2 Standard encode
//! (field-split per §7.5.3, two pictures sharing one frame_header()),
//! and an 8-bit 4:4:4 + alpha 4444 encode (the §5.3.3 / §7.1.2
//! lossless alpha coder on top of the ap4h colour baseline). A deep
//! alpha-typed case (12-bit 4:4:4 + 12-bit alpha through the
//! `Yuva444P12Le` registry surface) adds the §7.5.2-mirror 12→16
//! alpha promotion and the 16-bit-word colour reads.
//!
//! Every input is synthesized in-process, so the bench needs no
//! external fixtures.
//!
//! Run with e.g.:
//! `cargo bench --bench encode -- --warm-up-time 1 --measurement-time 3`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};

use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::{
    encode_frame_interlaced, encode_frame_with_alpha, encode_frame_with_depth, make_encoder,
};
use oxideav_prores::frame::{ChromaFormat, Profile};

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

/// 12-bit 4:4:4 gradient: each sample is packed little-endian into two
/// bytes, valid range `0..=4095`.
fn source_444_12bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; w * h * 2];
    let mut cr = vec![0u8; w * h * 2];
    let put = |buf: &mut [u8], idx: usize, v: u16| {
        buf[idx * 2..idx * 2 + 2].copy_from_slice(&v.to_le_bytes());
    };
    for j in 0..h {
        for i in 0..w {
            let v: u16 = ((i * 29 + j * 17) as u16 % 4096).min(4095);
            put(&mut y, j * w + i, v);
            let cb_v = (2048 + ((i as i32 - w as i32 / 2) * 16).clamp(-1024, 1024)) as u16;
            let cr_v = (2048 + ((j as i32 - h as i32 / 2) * 16).clamp(-1024, 1024)) as u16;
            put(&mut cb, j * w + i, cb_v);
            put(&mut cr, j * w + i, cr_v);
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
                stride: w * 2,
                data: cb,
            },
            VideoPlane {
                stride: w * 2,
                data: cr,
            },
        ],
    }
}

fn bench_encode(c: &mut Criterion) {
    let src_422 = source_422_8bit(W, H);
    let src_444 = source_444_8bit(W, H);
    let src_422_10 = source_422_10bit(W, H);

    let mut group = c.benchmark_group("encode_frame");
    // Throughput is measured in pixels (W*H luma samples) per second, so
    // 4:2:2 and 4:4:4 cases are directly comparable on the luma plane.
    group.throughput(Throughput::Elements((W * H) as u64));

    // Four 4:2:2 profiles on 8-bit 4:2:2 input, each at its default qi.
    for profile in [Profile::Proxy, Profile::Lt, Profile::Standard, Profile::Hq] {
        let qi = profile.default_quant_index();
        let name = format!("{}_422_8bit_128x96", fourcc_str(profile));
        group.bench_function(name, |b| {
            b.iter(|| {
                encode_frame_with_depth(
                    black_box(&src_422),
                    W,
                    H,
                    ChromaFormat::Y422,
                    BitDepth::Eight,
                    profile,
                    qi,
                )
                .expect("encode 422")
            });
        });
    }

    // Two 4:4:4 profiles on 8-bit 4:4:4 input, each at its default qi.
    for profile in [Profile::Prores4444, Profile::Prores4444Xq] {
        let qi = profile.default_quant_index();
        let name = format!("{}_444_8bit_128x96", fourcc_str(profile));
        group.bench_function(name, |b| {
            b.iter(|| {
                encode_frame_with_depth(
                    black_box(&src_444),
                    W,
                    H,
                    ChromaFormat::Y444,
                    BitDepth::Eight,
                    profile,
                    qi,
                )
                .expect("encode 444")
            });
        });
    }

    // 10-bit 4:2:2 Standard — exercises the deeper-bit read path.
    group.bench_function("apcn_422_10bit_128x96", |b| {
        b.iter(|| {
            encode_frame_with_depth(
                black_box(&src_422_10),
                W,
                H,
                ChromaFormat::Y422,
                BitDepth::Ten,
                Profile::Standard,
                Profile::Standard.default_quant_index(),
            )
            .expect("encode 10-bit")
        });
    });

    // Interlaced 8-bit 4:2:2 Standard (top-field-first) — two pictures
    // sharing one frame_header(), field-split per §7.5.3.
    group.bench_function("apcn_422_8bit_interlaced_tff_128x96", |b| {
        b.iter(|| {
            encode_frame_interlaced(
                black_box(&src_422),
                W,
                H,
                ChromaFormat::Y422,
                BitDepth::Eight,
                Profile::Standard,
                Profile::Standard.default_quant_index(),
                None,
                1, // top-field-first
            )
            .expect("encode interlaced")
        });
    });

    // 8-bit 4:4:4 + diagonal 8-bit alpha gradient — the §5.3.3 / §7.1.2
    // lossless alpha coder rides every slice; same colour content as
    // the ap4h case above, so the delta isolates the alpha coding cost.
    let src_444_alpha = {
        let mut src = source_444_8bit(W, H);
        let (w, h) = (W as usize, H as usize);
        let mut a = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                a[j * w + i] = (((i + j) * 255) / (w + h - 2)) as u8;
            }
        }
        src.planes.push(VideoPlane { stride: w, data: a });
        src
    };
    group.bench_function("ap4h_444a_8bit_alpha_128x96", |b| {
        b.iter(|| {
            encode_frame_with_alpha(
                black_box(&src_444_alpha),
                W,
                H,
                ChromaFormat::Y444,
                BitDepth::Eight,
                Profile::Prores4444,
                Profile::Prores4444.default_quant_index(),
                Some(AlphaChannelType::Eight),
            )
            .expect("encode 444 + alpha")
        });
    });

    // 12-bit 4:4:4 + 12-bit alpha through the deep alpha-typed registry
    // surface — the §7.5.2-mirror 12→16 alpha promotion plus the
    // 16-bit-word colour reads, on top of the alpha coding cost.
    let src_444_alpha_deep = {
        let (w, h) = (W as usize, H as usize);
        let mut src = source_444_12bit(W, H);
        let mut a = vec![0u8; w * h * 2];
        for j in 0..h {
            for i in 0..w {
                let v = (((i + j) * 4095) / (w + h - 2)) as u16;
                a[(j * w + i) * 2..(j * w + i) * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        src.planes.push(VideoPlane {
            stride: w * 2,
            data: a,
        });
        src
    };
    let mut deep_params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    deep_params.media_type = MediaType::Video;
    deep_params.width = Some(W);
    deep_params.height = Some(H);
    deep_params.pixel_format = Some(PixelFormat::Yuva444P12Le);
    let mut deep_enc = make_encoder(&deep_params).expect("make_encoder Yuva444P12Le");
    group.bench_function("ap4h_444a_12bit_yuva_deep_128x96", |b| {
        b.iter(|| {
            deep_enc
                .send_frame(black_box(&Frame::Video(src_444_alpha_deep.clone())))
                .expect("send_frame deep typed");
            deep_enc
                .receive_packet()
                .expect("receive_packet deep typed")
        });
    });

    group.finish();
}

/// Lower-case fourcc string for use in bench ids.
fn fourcc_str(profile: Profile) -> String {
    String::from_utf8_lossy(profile.fourcc()).into_owned()
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
