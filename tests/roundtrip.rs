//! Integration: encode -> decode round-trip per profile.
//!
//! Each profile is exercised through the top-level codec registry so
//! this test also verifies the `pick_profile` bitrate selector. Every
//! profile must decode to the same dimensions as the source and reach
//! a reasonable luma PSNR on a smooth synthetic image.

use oxideav_core::frame::VideoPlane;
use oxideav_core::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, VideoFrame};

/// Source frame bundled with the stream-level metadata that used to
/// live on `VideoFrame` itself. Tests pass this pair around so the
/// roundtrip helper can stamp the matching `CodecParameters`.
struct Source {
    frame: VideoFrame,
    format: PixelFormat,
    width: u32,
    height: u32,
}

fn synthetic_422(width: u32, height: u32) -> Source {
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
    Source {
        frame: VideoFrame {
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
        },
        format: PixelFormat::Yuv422P,
        width,
        height,
    }
}

fn synthetic_444(width: u32, height: u32) -> Source {
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
    Source {
        frame: VideoFrame {
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
        },
        format: PixelFormat::Yuv444P,
        width,
        height,
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

fn roundtrip(source: &Source, bit_rate: Option<u64>, expected_fourcc: &[u8; 4], min_psnr: f64) {
    let mut enc_params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    enc_params.media_type = MediaType::Video;
    enc_params.width = Some(source.width);
    enc_params.height = Some(source.height);
    enc_params.pixel_format = Some(source.format);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    enc_params.bit_rate = bit_rate;

    let mut reg = CodecRegistry::new();
    oxideav_prores::register(&mut reg);

    let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
    encoder
        .send_frame(&Frame::Video(source.frame.clone()))
        .expect("send_frame");
    let pkt = encoder.receive_packet().expect("receive_packet");

    // RDD 36 frames don't carry a profile code (it lives at the container
    // level via FourCC). Verify the encoder's profile *selection* via the
    // public `pick_profile` helper instead — that's what produced this packet.
    let chroma = match source.format {
        PixelFormat::Yuv422P => oxideav_prores::frame::ChromaFormat::Y422,
        PixelFormat::Yuv444P => oxideav_prores::frame::ChromaFormat::Y444,
        _ => panic!("unexpected pixel format in test"),
    };
    let chosen = oxideav_prores::encoder::pick_profile(chroma, bit_rate);
    assert_eq!(
        chosen.fourcc(),
        expected_fourcc,
        "pick_profile({chroma:?}, {bit_rate:?}) has fourcc {:?}, want {:?}",
        std::str::from_utf8(chosen.fourcc()).unwrap_or("?"),
        std::str::from_utf8(expected_fourcc).unwrap_or("?"),
    );
    // Sanity: the produced packet does parse as a valid RDD 36 frame
    // and the chroma_format matches.
    let (fh, _) = oxideav_prores::frame::parse_frame(&pkt.data).expect("parse_frame");
    assert_eq!(fh.chroma_format, chroma);

    let mut decoder = reg.make_decoder(&enc_params).expect("make_decoder");
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let decoded = match frame {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };

    // Stream-level format / width / height live on CodecParameters now;
    // the frame just carries planes.
    assert_eq!(decoded.planes.len(), 3);

    let p = luma_psnr(&source.frame.planes[0].data, &decoded.planes[0].data);
    assert!(
        p > min_psnr,
        "luma PSNR {p:.2} dB under threshold {min_psnr:.2} dB for fourcc {}",
        std::str::from_utf8(expected_fourcc).unwrap_or("?")
    );
}

#[test]
fn roundtrip_422_proxy() {
    let src = synthetic_422(64, 48);
    // Low bit_rate selects Proxy (quant 8).
    roundtrip(&src, Some(40_000_000), b"apco", 28.0);
}

#[test]
fn roundtrip_422_lt() {
    let src = synthetic_422(64, 48);
    roundtrip(&src, Some(100_000_000), b"apcs", 30.0);
}

#[test]
fn roundtrip_422_standard() {
    let src = synthetic_422(64, 48);
    // Default (no bit_rate) is Standard.
    roundtrip(&src, None, b"apcn", 32.0);
}

#[test]
fn roundtrip_422_standard_via_bitrate() {
    let src = synthetic_422(64, 48);
    roundtrip(&src, Some(150_000_000), b"apcn", 32.0);
}

#[test]
fn roundtrip_422_hq() {
    let src = synthetic_422(64, 48);
    roundtrip(&src, Some(220_000_000), b"apch", 38.0);
}

#[test]
fn roundtrip_4444() {
    let src = synthetic_444(64, 48);
    roundtrip(&src, None, b"ap4h", 36.0);
}

#[test]
fn roundtrip_4444_xq() {
    let src = synthetic_444(64, 48);
    roundtrip(&src, Some(500_000_000), b"ap4x", 40.0);
}

#[test]
fn roundtrip_odd_dimensions_422() {
    // Non-multiple-of-16: exercises MB-grid padding + cropping.
    let src = synthetic_422(40, 24);
    roundtrip(&src, None, b"apcn", 30.0);
}

#[test]
fn roundtrip_odd_dimensions_444() {
    let src = synthetic_444(40, 24);
    roundtrip(&src, None, b"ap4h", 34.0);
}

// ────────────────── 10-bit (Yuv422P10Le / Yuv444P10Le) ──────────────────

/// Build a 10-bit 4:2:2 source: each plane is `samples_per_row * 2` bytes
/// wide, samples packed little-endian, valid range `0..=1023`. The
/// luma gradient sweeps the full 10-bit range so the round-trip exercises
/// the high bits — an 8-bit-only emit path would clip everything above
/// 1020 and lose the top end of the sweep.
fn synthetic_422_10bit(width: u32, height: u32) -> Source {
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
    Source {
        frame: VideoFrame {
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
        },
        format: PixelFormat::Yuv422P10Le,
        width,
        height,
    }
}

/// 10-bit equivalent of [`luma_psnr`]. Operates on packed-LE u16 data.
fn luma_psnr_10bit(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 2, 0);
    let mut mse = 0.0f64;
    let n = a.len() / 2;
    for i in 0..n {
        let av = u16::from_le_bytes([a[i * 2], a[i * 2 + 1]]) as f64;
        let bv = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) as f64;
        let d = av - bv;
        mse += d * d;
    }
    mse /= n as f64;
    if mse == 0.0 {
        return 120.0;
    }
    10.0 * (1023.0_f64 * 1023.0 / mse).log10()
}

fn roundtrip_10bit(
    source: &Source,
    bit_rate: Option<u64>,
    expected_fourcc: &[u8; 4],
    min_psnr: f64,
) {
    let mut enc_params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    enc_params.media_type = MediaType::Video;
    enc_params.width = Some(source.width);
    enc_params.height = Some(source.height);
    enc_params.pixel_format = Some(source.format);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    enc_params.bit_rate = bit_rate;

    let mut reg = CodecRegistry::new();
    oxideav_prores::register(&mut reg);

    let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
    encoder
        .send_frame(&Frame::Video(source.frame.clone()))
        .expect("send_frame");
    let pkt = encoder.receive_packet().expect("receive_packet");

    let chroma = match source.format {
        PixelFormat::Yuv422P10Le => oxideav_prores::frame::ChromaFormat::Y422,
        PixelFormat::Yuv444P10Le => oxideav_prores::frame::ChromaFormat::Y444,
        _ => panic!("unexpected pixel format in 10-bit test"),
    };
    let chosen = oxideav_prores::encoder::pick_profile(chroma, bit_rate);
    assert_eq!(chosen.fourcc(), expected_fourcc);
    let (fh, _) = oxideav_prores::frame::parse_frame(&pkt.data).expect("parse_frame");
    assert_eq!(fh.chroma_format, chroma);

    let mut decoder = reg.make_decoder(&enc_params).expect("make_decoder");
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let decoded = match frame {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };
    assert_eq!(decoded.planes.len(), 3);

    // 10-bit planes are 2x the byte width of 8-bit ones.
    assert_eq!(
        decoded.planes[0].data.len(),
        source.frame.planes[0].data.len(),
        "10-bit luma plane size mismatch"
    );
    let p = luma_psnr_10bit(&source.frame.planes[0].data, &decoded.planes[0].data);
    eprintln!(
        "10-bit luma PSNR for fourcc {} = {p:.2} dB",
        std::str::from_utf8(expected_fourcc).unwrap_or("?"),
    );
    assert!(
        p > min_psnr,
        "10-bit luma PSNR {p:.2} dB under threshold {min_psnr:.2} dB for fourcc {}",
        std::str::from_utf8(expected_fourcc).unwrap_or("?"),
    );
}

#[test]
fn roundtrip_422_10bit_standard() {
    let src = synthetic_422_10bit(64, 48);
    roundtrip_10bit(&src, None, b"apcn", 32.0);
}

#[test]
fn roundtrip_422_10bit_hq() {
    let src = synthetic_422_10bit(64, 48);
    roundtrip_10bit(&src, Some(220_000_000), b"apch", 38.0);
}

/// 10-bit decoder honors caller's pixel_format choice — same source bits
/// fed back as `Yuv422P10Le` planes (LE u16 pairs) instead of single-byte
/// 8-bit planes. Verifies the byte stride and dynamic range.
#[test]
fn roundtrip_422_10bit_dynamic_range() {
    let src = synthetic_422_10bit(64, 48);
    let mut enc_params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    enc_params.media_type = MediaType::Video;
    enc_params.width = Some(src.width);
    enc_params.height = Some(src.height);
    enc_params.pixel_format = Some(PixelFormat::Yuv422P10Le);
    let mut reg = CodecRegistry::new();
    oxideav_prores::register(&mut reg);
    let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
    encoder
        .send_frame(&Frame::Video(src.frame.clone()))
        .expect("send_frame");
    let pkt = encoder.receive_packet().expect("receive_packet");
    let mut decoder = reg.make_decoder(&enc_params).expect("make_decoder");
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let decoded = match frame {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };
    // Luma plane is `width * 2` bytes per row.
    assert_eq!(decoded.planes[0].stride, (src.width as usize) * 2);
    // The decoded luma should span more than 8 bits of dynamic range —
    // i.e., contain at least one value > 255 — proving the 10-bit emit
    // path actually carries the high bits.
    let lum = &decoded.planes[0].data;
    let mut over_255 = false;
    for chunk in lum.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v > 255 {
            over_255 = true;
            break;
        }
    }
    assert!(
        over_255,
        "decoded 10-bit luma never exceeds 255 — emit path is collapsing to 8 bits"
    );
}
