//! Two-pass per-frame rate-control integration tests.
//!
//! Validates [`EncoderConfig::with_rate_control`]: given a `bit_rate` and
//! `frame_rate`, each encoded frame lands within ±5 % of the nominal
//! per-frame byte target (`RATE_CTRL_TOLERANCE`), provided the target is
//! within the achievable range [min_frame, max_frame] for the resolution.
//!
//! Methodology: encode twice at known qi values (lo and hi) to bracket the
//! achievable range, then set the target to the geometric mean — a point
//! the binary search can reliably reach in ≤ 10 passes.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, VideoFrame};
use oxideav_prores::encoder::{
    encode_frame_with_depth, make_encoder_with_config, EncoderConfig, RATE_CTRL_TOLERANCE,
};
use oxideav_prores::frame::{ChromaFormat, Profile};
use oxideav_prores::CODEC_ID_STR;

fn synth_422(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    let mut rng: u32 = 0xDEAD_BEEF;
    let mut next = || -> u8 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 24) as u8
    };
    for j in 0..h {
        let base = (j * 200 / h) as u8;
        for i in 0..w {
            let n = next();
            y[j * w + i] = base.wrapping_add(n / 4);
        }
        for i in 0..cw {
            cb[j * cw + i] = 128u8.wrapping_add(next() / 8);
            cr[j * cw + i] = 128u8.wrapping_add(next() / 8);
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

fn synth_444(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    let mut rng: u32 = 0xCAFE_BABE;
    let mut next = || -> u8 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 24) as u8
    };
    for j in 0..h {
        let base = (j * 200 / h) as u8;
        for i in 0..w {
            let n = next();
            y[j * w + i] = base.wrapping_add(n / 4);
            cb[j * w + i] = 128u8.wrapping_add(next() / 8);
            cr[j * w + i] = 128u8.wrapping_add(next() / 8);
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

/// Encode one frame with rate control and verify the encoded size lands
/// within [`RATE_CTRL_TOLERANCE`] of `target_bytes`.
fn assert_rate_ctrl_hits_target(
    frame: VideoFrame,
    pix: PixelFormat,
    width: u32,
    height: u32,
    target_bytes: usize,
    fps_num: i64,
    fps_den: i64,
) {
    let tol_lo = (target_bytes as f64 * (1.0 - RATE_CTRL_TOLERANCE)) as usize;
    let tol_hi = (target_bytes as f64 * (1.0 + RATE_CTRL_TOLERANCE)) as usize;
    // Set bit_rate so that bytes_per_frame == target_bytes at fps.
    let bit_rate = (target_bytes as u64 * 8 * fps_num as u64) / fps_den as u64;

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pix);
    params.bit_rate = Some(bit_rate);
    params.frame_rate = Some(Rational::new(fps_num, fps_den));

    let cfg = EncoderConfig::default().with_rate_control();
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    let actual = pkt.data.len();

    eprintln!(
        "rate_ctrl: target={target_bytes} B [{tol_lo}..{tol_hi}], actual={actual} B, \
         err={:+.1}%, bit_rate={bit_rate} bps",
        100.0 * (actual as f64 - target_bytes as f64) / target_bytes as f64,
    );
    assert!(
        actual >= tol_lo && actual <= tol_hi,
        "rate control missed target: actual={actual} B not in [{tol_lo}, {tol_hi}] \
         (target={target_bytes} B, ±{:.0}%)",
        RATE_CTRL_TOLERANCE * 100.0,
    );
}

/// Probe the achievable frame-size range for a given source + chroma at
/// two anchor qi values. Returns `(large_size, small_size)` at
/// `(qi_lo, qi_hi)` — callers pick a target in between.
fn probe_bracket(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    profile: Profile,
    qi_lo: u8,
    qi_hi: u8,
) -> (usize, usize) {
    let large = encode_frame_with_depth(
        frame,
        width,
        height,
        chroma,
        oxideav_prores::decoder::BitDepth::Eight,
        profile,
        qi_lo,
    )
    .expect("probe encode lo")
    .len();
    let small = encode_frame_with_depth(
        frame,
        width,
        height,
        chroma,
        oxideav_prores::decoder::BitDepth::Eight,
        profile,
        qi_hi,
    )
    .expect("probe encode hi")
    .len();
    (large, small)
}

// ──────────────── 422 profiles ────────────────────────────────────────────────

/// Rate control for 422 Standard profile at 64×48. The target is set to
/// the geometric midpoint between qi=2 (high quality) and qi=12 (low
/// quality) frame sizes — guaranteed to be inside the achievable range.
#[test]
fn rate_ctrl_422_standard_64x48_midpoint() {
    let width = 64u32;
    let height = 48u32;
    let src = synth_422(width, height);
    let (large, small) = probe_bracket(
        &src,
        width,
        height,
        ChromaFormat::Y422,
        Profile::Standard,
        2,
        16,
    );
    eprintln!("422 Standard 64x48: qi=2 → {large} B, qi=16 → {small} B");
    // Geometric mean of the bracket as target.
    let target = ((large as f64 * small as f64).sqrt()) as usize;
    assert_rate_ctrl_hits_target(src, PixelFormat::Yuv422P, width, height, target, 30, 1);
}

/// Rate control for 422 HQ profile at 64×48.
#[test]
fn rate_ctrl_422_hq_64x48_midpoint() {
    let width = 64u32;
    let height = 48u32;
    let src = synth_422(width, height);
    let (large, small) = probe_bracket(&src, width, height, ChromaFormat::Y422, Profile::Hq, 2, 16);
    eprintln!("422 HQ 64x48: qi=2 → {large} B, qi=16 → {small} B");
    let target = ((large as f64 * small as f64).sqrt()) as usize;
    assert_rate_ctrl_hits_target(src, PixelFormat::Yuv422P, width, height, target, 30, 1);
}

/// Rate control for 422 Proxy profile at 64×48.
#[test]
fn rate_ctrl_422_proxy_64x48_midpoint() {
    let width = 64u32;
    let height = 48u32;
    let src = synth_422(width, height);
    let (large, small) = probe_bracket(
        &src,
        width,
        height,
        ChromaFormat::Y422,
        Profile::Proxy,
        2,
        16,
    );
    eprintln!("422 Proxy 64x48: qi=2 → {large} B, qi=16 → {small} B");
    let target = ((large as f64 * small as f64).sqrt()) as usize;
    assert_rate_ctrl_hits_target(src, PixelFormat::Yuv422P, width, height, target, 30, 1);
}

// ──────────────── 4444 profile ────────────────────────────────────────────────

#[test]
fn rate_ctrl_4444_64x48_midpoint() {
    let width = 64u32;
    let height = 48u32;
    let src = synth_444(width, height);
    let (large, small) = probe_bracket(
        &src,
        width,
        height,
        ChromaFormat::Y444,
        Profile::Prores4444,
        2,
        16,
    );
    eprintln!("4444 64x48: qi=2 → {large} B, qi=16 → {small} B");
    let target = ((large as f64 * small as f64).sqrt()) as usize;
    assert_rate_ctrl_hits_target(src, PixelFormat::Yuv444P, width, height, target, 30, 1);
}

// ──────────────── Larger resolution: 128×96 ───────────────────────────────────

#[test]
fn rate_ctrl_422_hq_128x96_midpoint() {
    let width = 128u32;
    let height = 96u32;
    let src = synth_422(width, height);
    let (large, small) = probe_bracket(&src, width, height, ChromaFormat::Y422, Profile::Hq, 2, 16);
    eprintln!("422 HQ 128x96: qi=2 → {large} B, qi=16 → {small} B");
    let target = ((large as f64 * small as f64).sqrt()) as usize;
    assert_rate_ctrl_hits_target(src, PixelFormat::Yuv422P, width, height, target, 30, 1);
}

#[test]
fn rate_ctrl_4444_128x96_midpoint() {
    let width = 128u32;
    let height = 96u32;
    let src = synth_444(width, height);
    let (large, small) = probe_bracket(
        &src,
        width,
        height,
        ChromaFormat::Y444,
        Profile::Prores4444,
        2,
        16,
    );
    eprintln!("4444 128x96: qi=2 → {large} B, qi=16 → {small} B");
    let target = ((large as f64 * small as f64).sqrt()) as usize;
    assert_rate_ctrl_hits_target(src, PixelFormat::Yuv444P, width, height, target, 30, 1);
}

// ──────────────── Rate control does not regress single-pass when disabled ─────

#[test]
fn rate_ctrl_disabled_decodes_cleanly() {
    // Without rate_control the encoder must produce a valid decodable stream.
    let width = 64u32;
    let height = 48u32;
    let src = synth_422(width, height);

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    params.bit_rate = Some(180_000_000);
    params.frame_rate = Some(Rational::new(30, 1));

    let cfg_off = EncoderConfig::default();
    let mut enc = make_encoder_with_config(&params, cfg_off).expect("make_encoder");
    enc.send_frame(&Frame::Video(src.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    // Decode must succeed.
    let mut reg = oxideav_core::CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        oxideav_core::Frame::Video(v) => {
            assert_eq!(v.planes.len(), 3);
        }
        _ => panic!("expected video frame"),
    }
}

/// Rate control at an unreachable target (target > max achievable
/// frame size at qi=1) must not panic and must return a valid
/// (decodable) frame.
#[test]
fn rate_ctrl_unreachable_target_does_not_panic() {
    let width = 64u32;
    let height = 48u32;
    let src = synth_422(width, height);

    // Probe max size (at qi=1, finest possible)
    let max_size = encode_frame_with_depth(
        &src,
        width,
        height,
        ChromaFormat::Y422,
        oxideav_prores::decoder::BitDepth::Eight,
        Profile::Standard,
        1,
    )
    .expect("probe max")
    .len();
    // Target is 10× the max (physically unreachable)
    let target_bytes = max_size * 10;
    let bit_rate = (target_bytes as u64 * 8 * 30) as u64;

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    params.bit_rate = Some(bit_rate);
    params.frame_rate = Some(Rational::new(30, 1));

    let cfg = EncoderConfig::default().with_rate_control();
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(src.clone()))
        .expect("send_frame");
    let pkt = enc
        .receive_packet()
        .expect("no panic on unreachable target");

    // The returned frame must decode cleanly.
    let mut reg = oxideav_core::CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        oxideav_core::Frame::Video(v) => {
            assert_eq!(v.planes.len(), 3);
        }
        _ => panic!("expected video frame"),
    }
    eprintln!(
        "unreachable target={target_bytes} B, actual={} B (max achievable={max_size} B)",
        pkt.data.len()
    );
}
