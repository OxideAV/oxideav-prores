//! Alpha-channel coding through the high-level `Encoder` path (RDD 36
//! §5.3.3 + §7.1.2 + §6.1.1 Table 7).
//!
//! Before `EncoderConfig::alpha_channel_type` existed, alpha encoding
//! was reachable only through the free function
//! `encoder::encode_frame_with_alpha` — a registry-built encoder (or
//! any caller driving the `Encoder` trait) silently could NOT emit
//! alpha: `send_frame` hard-wired `alpha_channel_type = None`, so a
//! 4-plane input frame was refused with a plane-count error, and the
//! rate-control path could not carry alpha at all.
//!
//! These tests pin the closed gap end-to-end:
//!
//! * explicit `with_alpha_channel_type(Eight | Sixteen)` through
//!   `make_encoder_with_config` + `send_frame`,
//! * **auto-detection**: a plain `make_encoder` (registry-equivalent,
//!   no config at all) fed a 4-plane frame enables alpha by itself,
//!   inferring 8- vs 16-bit from the alpha plane's bytes-per-sample,
//! * alpha + two-pass rate control (every trial encode carries the
//!   lossless alpha blob; the frame still lands within tolerance),
//! * alpha + interlaced field-pair output through the config path,
//! * the §6.4 version rule: an alpha-bearing frame is bitstream
//!   version 1 even for 4:2:2 chroma,
//! * error paths: an explicit alpha request with a 3-plane frame, and
//!   an undetectable alpha stride under auto-detection.
//!
//! Alpha is coded losslessly per §7.1.2, so every roundtrip asserts
//! byte-exact alpha recovery, not a PSNR bar.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::{decode_packet, decode_packet_with_depth, BitDepth};
use oxideav_prores::encoder::{make_encoder, make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile};

const W: u32 = 64;
const H: u32 = 48;

fn params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("prores"));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

/// Deterministic 8-bit alpha plane: a diagonal gradient exercising the
/// Table 12 run codes and the Table 13 difference codes.
fn alpha_plane_8(w: usize, h: usize) -> VideoPlane {
    let mut a = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            a[j * w + i] = (((i + j) * 255) / (w + h - 2)) as u8;
        }
    }
    VideoPlane { stride: w, data: a }
}

/// Deterministic 16-bit LE alpha plane sweeping the full range.
fn alpha_plane_16(w: usize, h: usize) -> VideoPlane {
    let mut a = vec![0u8; w * h * 2];
    for j in 0..h {
        for i in 0..w {
            let v = (((i + j) * 65535) / (w + h - 2)) as u16;
            let off = (j * w + i) * 2;
            a[off] = (v & 0xFF) as u8;
            a[off + 1] = (v >> 8) as u8;
        }
    }
    VideoPlane {
        stride: w * 2,
        data: a,
    }
}

/// 8-bit 4:4:4 YCbCr content with non-trivial AC energy.
fn yuv444_8(w: usize, h: usize) -> Vec<VideoPlane> {
    let mut y = vec![0u8; w * h];
    let mut cb = vec![128u8; w * h];
    let mut cr = vec![128u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (32 + ((i * 3 + j * 2) % 180)) as u8;
            cb[j * w + i] = (128 + ((i as i32 - w as i32 / 2) / 2).clamp(-48, 48)) as u8;
            cr[j * w + i] = (128 + ((j as i32 - h as i32 / 2) / 2).clamp(-48, 48)) as u8;
        }
    }
    vec![
        VideoPlane { stride: w, data: y },
        VideoPlane {
            stride: w,
            data: cb,
        },
        VideoPlane {
            stride: w,
            data: cr,
        },
    ]
}

/// 8-bit 4:2:2 YCbCr content (half-width chroma).
fn yuv422_8(w: usize, h: usize) -> Vec<VideoPlane> {
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let cb = vec![120u8; cw * h];
    let cr = vec![136u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (24 + ((i * 5 + j * 3) % 200)) as u8;
        }
    }
    vec![
        VideoPlane { stride: w, data: y },
        VideoPlane {
            stride: cw,
            data: cb,
        },
        VideoPlane {
            stride: cw,
            data: cr,
        },
    ]
}

fn frame_with(planes: Vec<VideoPlane>) -> Frame {
    Frame::Video(VideoFrame {
        pts: Some(0),
        planes,
    })
}

/// Decode `pkt` at 8-bit output and assert the 4th plane matches
/// `expected` byte-for-byte (8-bit alpha at 8-bit output is the §7.5.2
/// identity case of the lossless §7.1.2 code).
fn assert_alpha8_exact(pkt: &[u8], expected: &[u8]) {
    let out = decode_packet(pkt, Some(0)).expect("decode_packet");
    assert_eq!(out.planes.len(), 4, "decoded frame must carry alpha");
    assert_eq!(
        out.planes[3].data, expected,
        "8-bit alpha must round-trip byte-exactly (lossless §7.1.2)"
    );
}

// ─────────────────── explicit config: 8-bit alpha, 4444 ───────────────────

#[test]
fn config_alpha8_4444_via_send_frame() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let expected_alpha = alpha.data.clone();
    planes.push(alpha);

    let cfg = EncoderConfig::for_profile(Profile::Prores4444)
        .with_alpha_channel_type(AlphaChannelType::Eight);
    let mut enc =
        make_encoder_with_config(&params(PixelFormat::Yuv444P), cfg).expect("make encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 1, "Table 7 code for 8-bit alpha");
    assert_eq!(fh.bitstream_version, 1, "alpha requires version 1 (§6.4)");
    assert_eq!(fh.picture_count(), 1);
    assert_alpha8_exact(&pkt.data, &expected_alpha);
}

// ─────────────────── explicit config: 16-bit alpha, 4444 XQ ───────────────────

#[test]
fn config_alpha16_4444xq_via_send_frame() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_16(w, h);
    let expected_alpha = alpha.data.clone();
    planes.push(alpha);

    let cfg = EncoderConfig::for_profile(Profile::Prores4444Xq)
        .with_alpha_channel_type(AlphaChannelType::Sixteen);
    let mut enc =
        make_encoder_with_config(&params(PixelFormat::Yuv444P), cfg).expect("make encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 2, "Table 7 code for 16-bit alpha");
    assert_eq!(fh.bitstream_version, 1);

    // 16-bit coded alpha decoded at 12-bit output follows the §7.5.2
    // demote formula round(4095 * a / 65535); the coded stream itself
    // is lossless, so re-deriving the expectation from the source is
    // exact.
    let out = decode_packet_with_depth(
        &pkt.data,
        Some(0),
        Some((BitDepth::Twelve, ChromaFormat::Y444)),
    )
    .expect("decode 12-bit");
    assert_eq!(out.planes.len(), 4);
    for (i, chunk) in out.planes[3].data.chunks_exact(2).enumerate() {
        let got = u16::from_le_bytes([chunk[0], chunk[1]]);
        let src = u16::from_le_bytes([expected_alpha[i * 2], expected_alpha[i * 2 + 1]]);
        let want = ((src as u64 * 4095 + 32767) / 65535) as u16;
        assert_eq!(got, want, "alpha sample {i}: §7.5.2 demote mismatch");
    }
}

// ─────────────────── auto-detection (registry-equivalent path) ───────────────────

#[test]
fn plain_make_encoder_autodetects_8bit_alpha() {
    // No EncoderConfig at all — the exact construction the codec
    // registry performs. A 4-plane frame with a 1-byte-per-sample
    // alpha plane must enable 8-bit alpha automatically.
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let expected_alpha = alpha.data.clone();
    planes.push(alpha);

    let mut enc = make_encoder(&params(PixelFormat::Yuv444P)).expect("make_encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 1, "auto-detected 8-bit alpha");
    assert_alpha8_exact(&pkt.data, &expected_alpha);
}

#[test]
fn plain_make_encoder_autodetects_16bit_alpha() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    planes.push(alpha_plane_16(w, h));

    let mut enc = make_encoder(&params(PixelFormat::Yuv444P)).expect("make_encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 2, "auto-detected 16-bit alpha");
}

#[test]
fn plain_make_encoder_three_planes_stays_alpha_free() {
    // The auto-detection must not disturb the alpha-free path: a
    // 3-plane frame still emits alpha_channel_type = 0 and (4:2:2)
    // bitstream version 0.
    let (w, h) = (W as usize, H as usize);
    let mut enc = make_encoder(&params(PixelFormat::Yuv422P)).expect("make_encoder");
    enc.send_frame(&frame_with(yuv422_8(w, h)))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 0);
    assert_eq!(fh.bitstream_version, 0, "4:2:2 no-alpha stays version 0");
}

// ─────────────────── alpha + rate control ───────────────────

#[test]
fn rate_control_carries_alpha() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let expected_alpha = alpha.data.clone();
    planes.push(alpha);

    // 700 kbit/s at 25 fps → 3 500 bytes/frame, inside the achievable
    // band for this 64×48 content (≈4.3 KB at qi = 1 down to the
    // lossless-alpha floor at qi = 224).
    let mut p = params(PixelFormat::Yuv444P);
    p.bit_rate = Some(700_000);
    p.frame_rate = Some(oxideav_core::Rational::new(25, 1));
    let cfg = EncoderConfig::for_profile(Profile::Prores4444)
        .with_alpha_channel_type(AlphaChannelType::Eight)
        .with_rate_control();
    let mut enc = make_encoder_with_config(&p, cfg).expect("make encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(
        fh.alpha_channel_type, 1,
        "rate-controlled frame must still carry alpha"
    );
    // Alpha is lossless regardless of what qi the rate search picked.
    assert_alpha8_exact(&pkt.data, &expected_alpha);
    // 700 kbit/s at 25 fps → 3 500 bytes/frame target; the search must
    // land in the ±5 % band (the content is comfortably compressible
    // in both directions at this size).
    let target = 700_000usize / 8 / 25;
    let lo = target * 95 / 100;
    let hi = target * 105 / 100;
    assert!(
        (lo..=hi).contains(&pkt.data.len()),
        "rate-controlled alpha frame {} bytes outside [{lo}, {hi}]",
        pkt.data.len()
    );
}

// ─────────────────── alpha + interlaced via config ───────────────────

#[test]
fn config_alpha_interlaced_field_pair() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let expected_alpha = alpha.data.clone();
    planes.push(alpha);

    let cfg = EncoderConfig::for_profile(Profile::Prores4444)
        .with_alpha_channel_type(AlphaChannelType::Eight)
        .with_interlace_mode(1);
    let mut enc =
        make_encoder_with_config(&params(PixelFormat::Yuv444P), cfg).expect("make encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.interlace_mode, 1, "TFF");
    assert_eq!(fh.picture_count(), 2, "field pair");
    assert_eq!(fh.alpha_channel_type, 1);
    // The §7.5.3 deinterleave must reassemble the alpha plane exactly.
    assert_alpha8_exact(&pkt.data, &expected_alpha);
}

// ─────────────────── alpha on 4:2:2 forces version 1 ───────────────────

#[test]
fn alpha_on_422_is_version_1() {
    // RDD 36 §6.4: version 0 requires 4:2:2 AND no alpha. Alpha with
    // 4:2:2 chroma is representable on version 1 — the encoder must
    // raise the version rather than emit an illegal v0 stream.
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv422_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let expected_alpha = alpha.data.clone();
    planes.push(alpha);

    let cfg =
        EncoderConfig::for_profile(Profile::Hq).with_alpha_channel_type(AlphaChannelType::Eight);
    let mut enc =
        make_encoder_with_config(&params(PixelFormat::Yuv422P), cfg).expect("make encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 1);
    assert_eq!(
        fh.bitstream_version, 1,
        "4:2:2 + alpha must be bitstream_version 1 (§6.4)"
    );
    assert_alpha8_exact(&pkt.data, &expected_alpha);
}

// ─────────────────── error paths ───────────────────

#[test]
fn explicit_alpha_with_three_planes_is_error() {
    let (w, h) = (W as usize, H as usize);
    let cfg = EncoderConfig::for_profile(Profile::Prores4444)
        .with_alpha_channel_type(AlphaChannelType::Eight);
    let mut enc =
        make_encoder_with_config(&params(PixelFormat::Yuv444P), cfg).expect("make encoder");
    let err = enc
        .send_frame(&frame_with(yuv444_8(w, h)))
        .expect_err("3-plane frame with an explicit alpha request must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("planes"),
        "error must name the plane-count mismatch, got: {msg}"
    );
}

#[test]
fn undetectable_alpha_stride_is_error() {
    // 4-plane frame whose alpha stride is 3 bytes/sample — neither
    // 8-bit nor 16-bit LE. Auto-detection must refuse rather than
    // guess.
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    planes.push(VideoPlane {
        stride: w * 3,
        data: vec![0u8; w * 3 * h],
    });
    let mut enc = make_encoder(&params(PixelFormat::Yuv444P)).expect("make_encoder");
    let err = enc
        .send_frame(&frame_with(planes))
        .expect_err("3-bytes-per-sample alpha stride must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("alpha"),
        "error must mention alpha detection, got: {msg}"
    );
}
