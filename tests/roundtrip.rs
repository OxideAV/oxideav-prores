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

// ────────────────── 12-bit (Yuv422P12Le / Yuv444P12Le) ──────────────────

fn synthetic_444_12bit(width: u32, height: u32) -> Source {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; w * h * 2];
    let mut cr = vec![0u8; w * h * 2];
    for j in 0..h {
        for i in 0..w {
            // Sweep the full 12-bit range so the round-trip exercises bits 10/11.
            let v: u16 = ((i * 31 + j * 23) as u16 % 4096).min(4095);
            let off = (j * w + i) * 2;
            y[off] = (v & 0xFF) as u8;
            y[off + 1] = (v >> 8) as u8;
            let cb_v: u16 = (2048 + ((i as i32 - w as i32 / 2) * 16).clamp(-1024, 1024)) as u16;
            let cr_v: u16 = (2048 + ((j as i32 - h as i32 / 2) * 16).clamp(-1024, 1024)) as u16;
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
                    stride: w * 2,
                    data: cb,
                },
                VideoPlane {
                    stride: w * 2,
                    data: cr,
                },
            ],
        },
        format: PixelFormat::Yuv444P12Le,
        width,
        height,
    }
}

/// PSNR over packed-LE u16 samples in `0..=4095`.
fn psnr_12bit(a: &[u8], b: &[u8]) -> f64 {
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
    10.0 * (4095.0_f64 * 4095.0 / mse).log10()
}

#[test]
fn roundtrip_4444_12bit() {
    let src = synthetic_444_12bit(64, 48);
    let mut enc_params = CodecParameters::video(CodecId::new(oxideav_prores::CODEC_ID_STR));
    enc_params.media_type = MediaType::Video;
    enc_params.width = Some(src.width);
    enc_params.height = Some(src.height);
    enc_params.pixel_format = Some(PixelFormat::Yuv444P12Le);
    enc_params.frame_rate = Some(Rational::new(30, 1));

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
    assert_eq!(decoded.planes.len(), 3);
    assert_eq!(decoded.planes[0].stride, (src.width as usize) * 2);

    let p = psnr_12bit(&src.frame.planes[0].data, &decoded.planes[0].data);
    eprintln!("12-bit 4444 luma PSNR = {p:.2} dB");
    assert!(p > 32.0, "12-bit 4444 luma PSNR {p:.2} dB under threshold");

    // Confirm the 12-bit emit path actually carries values above the
    // 10-bit ceiling (1023). A broken path that collapsed to 10-bit
    // would never produce a sample > 1023.
    let mut max_v: u16 = 0;
    for chunk in decoded.planes[0].data.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v > max_v {
            max_v = v;
        }
    }
    assert!(
        max_v > 1023,
        "12-bit luma never exceeds 1023 ({max_v}) — emit path collapsed"
    );
    assert!(max_v <= 4095, "12-bit luma value out of range: {max_v}");
}

// ──────────────── 4444 + alpha (Yuv444P + 4th plane) ────────────────

/// 4444 source with an explicit alpha plane appended as the 4th
/// `VideoPlane`. The decoder treats `alpha_channel_type != 0` as a
/// signal to populate a 4th output plane; the encoder accepts the same
/// shape on input.
fn synthetic_444_with_alpha_8bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    let mut a = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 3 + j * 2) as u16 % 256) as u8;
            cb[j * w + i] = (128 + ((i as i32 - w as i32 / 2).clamp(-64, 64))) as u8;
            cr[j * w + i] = (128 + ((j as i32 - h as i32 / 2).clamp(-64, 64))) as u8;
            // Smooth diagonal alpha gradient — 0 at (0,0), 255 at (w-1,h-1).
            let n = (i + j) as u32 * 255 / (w + h - 2) as u32;
            a[j * w + i] = n as u8;
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
            VideoPlane { stride: w, data: a },
        ],
    }
}

/// Encode a 4444 frame whose 4th plane is alpha, decode it back, and
/// confirm both the YUV planes and the alpha plane round-trip.
///
/// This exercises the full alpha pipeline:
///   * encoder serialises an `alpha_channel_type=1` frame header
///   * encoder emits per-slice scanned_alpha() blobs
///   * decoder parses the alpha blob and produces a 4th plane
#[test]
fn roundtrip_4444_with_alpha_8bit() {
    let frame = synthetic_444_with_alpha_8bit(64, 48);

    // Encode the 4-plane frame as ap4h with 8-bit alpha
    // (alpha_channel_type=1).
    let pkt = oxideav_prores::encoder::encode_frame_with_alpha(
        &frame,
        64,
        48,
        oxideav_prores::frame::ChromaFormat::Y444,
        oxideav_prores::decoder::BitDepth::Eight,
        oxideav_prores::frame::Profile::Prores4444,
        2,
        Some(oxideav_prores::alpha::AlphaChannelType::Eight),
    )
    .expect("encode_frame_with_alpha");

    // Decode at 8-bit.
    let decoded = oxideav_prores::decoder::decode_packet_with_depth(
        &pkt,
        Some(0),
        Some((
            oxideav_prores::decoder::BitDepth::Eight,
            oxideav_prores::frame::ChromaFormat::Y444,
        )),
    )
    .expect("decode_packet_with_depth");

    assert_eq!(decoded.planes.len(), 4, "alpha plane missing in output");

    // Alpha is stored losslessly (entropy-coded run-length on raster
    // values, no DCT) — decoded alpha must match the source exactly.
    assert_eq!(
        decoded.planes[3].data, frame.planes[3].data,
        "alpha plane is not lossless"
    );

    // The YUV planes are lossy through DCT — assert non-trivial PSNR
    // so we know the encoder/decoder pair did something real.
    let p = luma_psnr(&frame.planes[0].data, &decoded.planes[0].data);
    eprintln!("4444+alpha luma PSNR = {p:.2} dB");
    assert!(p > 30.0, "4444+alpha luma PSNR too low: {p:.2}");
}

// ────────────────── interlaced (RDD 36 §5.1 + §7.5.3) ──────────────────

/// Build a 4:2:2 source whose field-row content differs from the
/// progressive-row content. A row-index modulated luma gradient lets us
/// verify the encoder + decoder split / interleave pixels at the
/// correct field offsets. A bug that swapped TFF and BFF would push
/// even/odd rows to the wrong field.
fn synthetic_422_field_distinct(width: u32, height: u32) -> Source {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![128u8; cw * h];
    let mut cr = vec![128u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            // Even rows = bright gradient, odd rows = dim — a bit-exact
            // round-trip preserves this; a swapped-field bug flips the
            // brightness pattern.
            let base = if j % 2 == 0 { 192 } else { 64 };
            y[j * w + i] = (base + (i as u16 % 32)) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-32, 32)) as u8;
            cr[j * cw + i] = (128 + (j as i32 - h as i32 / 2).clamp(-32, 32)) as u8;
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

fn roundtrip_interlaced_422(width: u32, height: u32, interlace_mode: u8, min_psnr: f64) {
    use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
    use oxideav_prores::encoder::encode_frame_interlaced;
    use oxideav_prores::frame::{ChromaFormat, Profile};

    let src = synthetic_422_field_distinct(width, height);
    let pkt = encode_frame_interlaced(
        &src.frame,
        width,
        height,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Hq,
        2,
        None,
        interlace_mode,
    )
    .expect("encode_frame_interlaced");

    let (fh, _) = oxideav_prores::frame::parse_frame(&pkt).expect("parse_frame");
    assert_eq!(
        fh.interlace_mode, interlace_mode,
        "frame header interlace_mode mismatch"
    );
    assert_eq!(
        fh.picture_count(),
        2,
        "interlaced frame must declare 2 pictures"
    );

    let decoded =
        decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Eight, ChromaFormat::Y422)))
            .expect("decode interlaced");
    assert_eq!(decoded.planes.len(), 3);
    assert_eq!(decoded.planes[0].data.len(), (width * height) as usize);

    let p = luma_psnr(&src.frame.planes[0].data, &decoded.planes[0].data);
    eprintln!("interlaced (mode={interlace_mode}) luma PSNR = {p:.2} dB");
    assert!(
        p > min_psnr,
        "interlaced luma PSNR {p:.2} dB under {min_psnr} dB"
    );

    // Per-field correctness: even-row luma should still average brighter
    // than odd-row luma after the round-trip (proves the field
    // assignment did not swap top + bottom).
    let mut even_sum = 0u64;
    let mut odd_sum = 0u64;
    let w = width as usize;
    for j in 0..(height as usize) {
        let row_sum: u64 = decoded.planes[0].data[j * w..(j + 1) * w]
            .iter()
            .map(|&v| v as u64)
            .sum();
        if j % 2 == 0 {
            even_sum += row_sum;
        } else {
            odd_sum += row_sum;
        }
    }
    assert!(
        even_sum > odd_sum,
        "interlaced field assignment swapped: even-row sum {even_sum} not > odd-row sum {odd_sum}"
    );
}

#[test]
fn roundtrip_interlaced_422_tff() {
    roundtrip_interlaced_422(64, 48, 1, 32.0);
}

#[test]
fn roundtrip_interlaced_422_bff() {
    roundtrip_interlaced_422(64, 48, 2, 32.0);
}

#[test]
fn roundtrip_interlaced_422_odd_height() {
    // odd vertical_size: top field gets ceil(h/2) rows, bottom floor.
    roundtrip_interlaced_422(64, 50, 1, 32.0);
}

#[test]
fn interlaced_frame_header_roundtrips() {
    use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
    use oxideav_prores::encoder::encode_frame_interlaced;
    use oxideav_prores::frame::{ChromaFormat, Profile};

    let src = synthetic_422_field_distinct(64, 48);
    for mode in [1u8, 2] {
        let pkt = encode_frame_interlaced(
            &src.frame,
            64,
            48,
            ChromaFormat::Y422,
            BitDepth::Eight,
            Profile::Standard,
            4,
            None,
            mode,
        )
        .expect("encode_frame_interlaced");
        let (fh, _) = oxideav_prores::frame::parse_frame(&pkt).expect("parse_frame");
        assert_eq!(fh.interlace_mode, mode);
        assert_eq!(fh.chroma_format, ChromaFormat::Y422);
        // Decode succeeds with the standard public path.
        let _frame =
            decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Eight, ChromaFormat::Y422)))
                .expect("decode_packet_with_depth");
    }
}
