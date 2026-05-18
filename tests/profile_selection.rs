//! Explicit `EncoderConfig::with_profile()` selection.
//!
//! Covers the encoder API addition that lets callers pin a specific
//! ProRes profile (Proxy / LT / Standard / HQ / 4444 / 4444 XQ) without
//! relying on the `bit_rate` → profile heuristic in
//! [`pick_profile`]. The profile chosen drives the default
//! quantisation index (RDD 36 §7.3 / Table 15) which in turn drives the
//! achievable packet-size range. The bitstream itself carries only the
//! `chroma_format` field per §5.1.1 — the profile is not signalled in
//! the wire format, so the override is a pure encoder-side
//! quality/rate-target adjustment.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::encoder::{make_encoder_with_config, pick_profile, EncoderConfig};
use oxideav_prores::frame::{ChromaFormat, Profile};
use oxideav_prores::{register_codecs, CODEC_ID_STR};

fn synth_422(width: u32, height: u32) -> VideoFrame {
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

fn synth_444(width: u32, height: u32) -> VideoFrame {
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

/// Encode one frame at the supplied profile, return the packet bytes.
fn encode_with_profile(
    frame: VideoFrame,
    pix: PixelFormat,
    width: u32,
    height: u32,
    profile: Profile,
) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pix);
    let cfg = EncoderConfig::default().with_profile(profile);
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    pkt.data
}

/// `with_profile(Proxy)` must yield a smaller frame than `with_profile(HQ)`
/// because the default quantisation index for Proxy (8) is coarser than HQ (2).
#[test]
fn with_profile_proxy_smaller_than_hq() {
    let width = 128u32;
    let height = 96u32;
    let proxy = encode_with_profile(
        synth_422(width, height),
        PixelFormat::Yuv422P,
        width,
        height,
        Profile::Proxy,
    );
    let hq = encode_with_profile(
        synth_422(width, height),
        PixelFormat::Yuv422P,
        width,
        height,
        Profile::Hq,
    );
    eprintln!(
        "Proxy: {} B, HQ: {} B (ratio HQ/Proxy = {:.2}×)",
        proxy.len(),
        hq.len(),
        hq.len() as f64 / proxy.len() as f64
    );
    assert!(
        hq.len() > proxy.len(),
        "HQ must be larger than Proxy (HQ={} B, Proxy={} B)",
        hq.len(),
        proxy.len(),
    );
}

/// `with_profile(Prores4444Xq)` must yield a larger frame than
/// `with_profile(Prores4444)` for the same 4:4:4 source — XQ default qi=1 is
/// finer than 4444 default qi=2.
#[test]
fn with_profile_xq_larger_than_4444() {
    let width = 128u32;
    let height = 96u32;
    let p4444 = encode_with_profile(
        synth_444(width, height),
        PixelFormat::Yuv444P,
        width,
        height,
        Profile::Prores4444,
    );
    let pxq = encode_with_profile(
        synth_444(width, height),
        PixelFormat::Yuv444P,
        width,
        height,
        Profile::Prores4444Xq,
    );
    eprintln!(
        "4444: {} B, 4444 XQ: {} B (ratio XQ/4444 = {:.2}×)",
        p4444.len(),
        pxq.len(),
        pxq.len() as f64 / p4444.len() as f64,
    );
    assert!(
        pxq.len() > p4444.len(),
        "XQ must be larger than 4444 (XQ={} B, 4444={} B)",
        pxq.len(),
        p4444.len(),
    );
}

/// `with_profile` for a 4:4:4 stream with `bit_rate < 400 Mbit/s` overrides
/// the `pick_profile` heuristic which would otherwise return 4444 (not XQ).
#[test]
fn with_profile_xq_overrides_low_bitrate_heuristic() {
    let width = 64u32;
    let height = 48u32;
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv444P);
    // bit_rate below the 400 Mbit/s XQ threshold: pick_profile -> Prores4444
    params.bit_rate = Some(100_000_000);
    assert_eq!(
        pick_profile(ChromaFormat::Y444, params.bit_rate),
        Profile::Prores4444,
        "sanity: pick_profile at 100 Mbit/s on Y444 returns 4444 (not XQ)",
    );
    // The override forces XQ regardless.
    let xq_bytes = encode_with_profile(
        synth_444(width, height),
        PixelFormat::Yuv444P,
        width,
        height,
        Profile::Prores4444Xq,
    );
    let default_bytes = {
        let cfg = EncoderConfig::default();
        let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
        enc.send_frame(&Frame::Video(synth_444(width, height)))
            .expect("send_frame");
        enc.receive_packet().expect("receive_packet").data
    };
    eprintln!(
        "Default-pick (4444 @ qi=2): {} B, with_profile(XQ @ qi=1): {} B",
        default_bytes.len(),
        xq_bytes.len()
    );
    assert!(
        xq_bytes.len() > default_bytes.len(),
        "with_profile(XQ) must produce a larger packet than the default pick_profile->4444 \
         choice at the same low bit_rate (XQ={} B, 4444={} B)",
        xq_bytes.len(),
        default_bytes.len(),
    );
}

/// Mismatch between override profile and pixel-format chroma is rejected.
#[test]
fn with_profile_chroma_mismatch_rejected() {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(48);
    // Y422 with Profile::Prores4444 (4:4:4) — mismatch.
    params.pixel_format = Some(PixelFormat::Yuv422P);
    let cfg = EncoderConfig::default().with_profile(Profile::Prores4444);
    let result = make_encoder_with_config(&params, cfg);
    let err = match result {
        Ok(_) => panic!("expected mismatch error, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("does not match") && msg.contains("Prores4444"),
        "expected chroma-mismatch error, got: {msg}",
    );
    // Y444 with Profile::Hq (4:2:2) — also mismatch.
    params.pixel_format = Some(PixelFormat::Yuv444P);
    let cfg = EncoderConfig::default().with_profile(Profile::Hq);
    let result = make_encoder_with_config(&params, cfg);
    let err = match result {
        Ok(_) => panic!("expected mismatch error, got Ok"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("does not match") && msg.contains("Hq"),
        "expected chroma-mismatch error, got: {msg}",
    );
}

/// `with_profile` composes with `with_quantization_index`: the explicit
/// qi overrides the per-profile default. (Validates that the profile
/// override only changes the seed/default, not the resulting bitstream
/// when the caller explicitly sets qi.)
#[test]
fn with_profile_and_explicit_qi_uses_qi() {
    let width = 64u32;
    let height = 48u32;
    // Same qi=4 forces the same packet size regardless of profile choice
    // (qi=4 is the SD default but a valid override for HQ as well).
    let src1 = synth_422(width, height);
    let src2 = synth_422(width, height);

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv422P);

    let mut enc1 = make_encoder_with_config(
        &params,
        EncoderConfig::default()
            .with_profile(Profile::Standard)
            .with_quantization_index(4),
    )
    .expect("std qi=4");
    enc1.send_frame(&Frame::Video(src1)).expect("send_frame");
    let pkt1 = enc1.receive_packet().expect("recv");

    let mut enc2 = make_encoder_with_config(
        &params,
        EncoderConfig::default()
            .with_profile(Profile::Hq)
            .with_quantization_index(4),
    )
    .expect("hq qi=4");
    enc2.send_frame(&Frame::Video(src2)).expect("send_frame");
    let pkt2 = enc2.receive_packet().expect("recv");

    // Standard and HQ are identical at the bitstream level (the chroma
    // format and bitstream syntax are identical; only the typical qi
    // differs by profile, and we pinned qi=4 on both). So the encoded
    // packets should be byte-identical.
    assert_eq!(
        pkt1.data, pkt2.data,
        "with_profile(Std) qi=4 and with_profile(Hq) qi=4 must produce the same bytes \
         (the profile only affects defaults — explicit qi overrides it)",
    );
}

/// Every output of `with_profile` is decodable end-to-end across all
/// six profiles (sanity).
#[test]
fn with_profile_all_six_decode_cleanly() {
    let width = 64u32;
    let height = 48u32;
    for &(profile, pix) in &[
        (Profile::Proxy, PixelFormat::Yuv422P),
        (Profile::Lt, PixelFormat::Yuv422P),
        (Profile::Standard, PixelFormat::Yuv422P),
        (Profile::Hq, PixelFormat::Yuv422P),
        (Profile::Prores4444, PixelFormat::Yuv444P),
        (Profile::Prores4444Xq, PixelFormat::Yuv444P),
    ] {
        let frame = if pix == PixelFormat::Yuv422P {
            synth_422(width, height)
        } else {
            synth_444(width, height)
        };
        let bytes = encode_with_profile(frame, pix, width, height, profile);

        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.media_type = MediaType::Video;
        params.width = Some(width);
        params.height = Some(height);
        params.pixel_format = Some(pix);
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let mut dec = reg.first_decoder(&params).expect("make_decoder");
        let pkt = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 30), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let f = dec.receive_frame().expect("receive_frame");
        match f {
            Frame::Video(v) => {
                assert!(
                    v.planes.len() >= 3,
                    "profile {profile:?} produced < 3 planes",
                );
            }
            _ => panic!("profile {profile:?} returned non-video frame"),
        }
        eprintln!("profile {profile:?} → packet {} B, decoded OK", 0);
    }
}
