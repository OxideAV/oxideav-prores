//! Integration: encode -> decode round-trip per profile.
//!
//! Each profile is exercised through the top-level codec registry so
//! this test also verifies the `pick_profile` bitrate selector. Every
//! profile must decode to the same dimensions as the source and reach
//! a reasonable luma PSNR on a smooth synthetic image.

use oxideav_core::frame::VideoPlane;
use oxideav_core::CodecRegistry;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, TimeBase, VideoFrame,
};

fn synthetic_422(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 3 + j * 2) as u16 % 256) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2).clamp(-64, 64))) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2).clamp(-64, 64))) as u8;
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv422P,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
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

fn synthetic_444(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 3 + j * 2) as u16 % 256) as u8;
            cb[j * w + i] = (128 + ((i as i32 - w as i32 / 2).clamp(-64, 64))) as u8;
            cr[j * w + i] = (128 + ((j as i32 - h as i32 / 2).clamp(-64, 64))) as u8;
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv444P,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
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

fn luma_psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut mse = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        mse += d * d;
    }
    mse /= a.len() as f64;
    if mse == 0.0 {
        return 120.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn roundtrip(
    original: &VideoFrame,
    bit_rate: Option<u64>,
    expected_fourcc: &[u8; 4],
    min_psnr: f64,
) {
    let width = original.width;
    let height = original.height;
    let pix = original.format;

    let mut enc_params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    enc_params.media_type = MediaType::Video;
    enc_params.width = Some(width);
    enc_params.height = Some(height);
    enc_params.pixel_format = Some(pix);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    enc_params.bit_rate = bit_rate;

    let mut reg = CodecRegistry::new();
    oxideav_prores::register(&mut reg);

    let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
    encoder
        .send_frame(&Frame::Video(original.clone()))
        .expect("send_frame");
    let pkt = encoder.receive_packet().expect("receive_packet");

    let (fh, _) = oxideav_prores::frame::parse_frame_header(&pkt.data).expect("parse_frame_header");
    assert_eq!(
        fh.profile.fourcc(),
        expected_fourcc,
        "profile selected for bit_rate={bit_rate:?} has fourcc {:?}, want {:?}",
        std::str::from_utf8(fh.profile.fourcc()).unwrap_or("?"),
        std::str::from_utf8(expected_fourcc).unwrap_or("?"),
    );

    let mut decoder = reg.make_decoder(&enc_params).expect("make_decoder");
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let decoded = match frame {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };

    assert_eq!(decoded.format, pix);
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.planes.len(), 3);

    let p = luma_psnr(&original.planes[0].data, &decoded.planes[0].data);
    assert!(
        p > min_psnr,
        "luma PSNR {p:.2} dB under threshold {min_psnr:.2} dB for fourcc {}",
        std::str::from_utf8(expected_fourcc).unwrap_or("?")
    );
}

#[test]
fn roundtrip_422_proxy() {
    let frame = synthetic_422(64, 48);
    // Low bit_rate selects Proxy (quant 8).
    roundtrip(&frame, Some(40_000_000), b"apco", 28.0);
}

#[test]
fn roundtrip_422_lt() {
    let frame = synthetic_422(64, 48);
    roundtrip(&frame, Some(100_000_000), b"apcs", 30.0);
}

#[test]
fn roundtrip_422_standard() {
    let frame = synthetic_422(64, 48);
    // Default (no bit_rate) is Standard.
    roundtrip(&frame, None, b"apcn", 32.0);
}

#[test]
fn roundtrip_422_standard_via_bitrate() {
    let frame = synthetic_422(64, 48);
    roundtrip(&frame, Some(150_000_000), b"apcn", 32.0);
}

#[test]
fn roundtrip_422_hq() {
    let frame = synthetic_422(64, 48);
    roundtrip(&frame, Some(220_000_000), b"apch", 38.0);
}

#[test]
fn roundtrip_4444() {
    let frame = synthetic_444(64, 48);
    roundtrip(&frame, None, b"ap4h", 36.0);
}

#[test]
fn roundtrip_4444_xq() {
    let frame = synthetic_444(64, 48);
    roundtrip(&frame, Some(500_000_000), b"ap4x", 40.0);
}

#[test]
fn roundtrip_odd_dimensions_422() {
    // Non-multiple-of-16: exercises MB-grid padding + cropping.
    let frame = synthetic_422(40, 24);
    roundtrip(&frame, None, b"apcn", 30.0);
}

#[test]
fn roundtrip_odd_dimensions_444() {
    let frame = synthetic_444(40, 24);
    roundtrip(&frame, None, b"ap4h", 34.0);
}
