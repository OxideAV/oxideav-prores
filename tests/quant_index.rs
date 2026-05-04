//! `EncoderConfig::quantization_index` integration tests.
//!
//! The encoder picks one `quantization_index` from the [`Profile`] (RDD
//! 36 §7.3 / Table 15) by default — `4` for 422 Standard, `8` for Proxy,
//! etc. Callers that want a different point on the rate/quality curve
//! without shifting profile (e.g. "give me HQ at qi=3" or "give me
//! Proxy at qi=12 to undercut the default Proxy bitrate") can override
//! the index via [`EncoderConfig::with_quantization_index`].
//!
//! What this driver checks:
//! * Lower `quantization_index` produces strictly larger packets at the
//!   same profile (qScale monotonicity in Table 15).
//! * Lower `quantization_index` produces strictly higher PSNR (uniform
//!   quantisation is consistent with smaller step → smaller error).
//! * Out-of-range indices (`0`, `225`) are rejected at encoder
//!   construction time.
//! * The override is honoured even when paired with a non-default
//!   profile, e.g. `Proxy` with `qi=2` produces a higher-quality (and
//!   larger) packet than the Proxy default `qi=8`.
//! * Bitstream is decode-clean (every per-slice qscale matches the
//!   override on the wire — verified indirectly by a roundtrip with the
//!   public registry path).

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::CODEC_ID_STR;

fn synth_textured(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    let mut rng: u32 = 0x1234_5678;
    let mut next_byte = || -> i32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 24) as i32) & 0xFF
    };
    for j in 0..h {
        for i in 0..w {
            let g = (i + j) * 200 / (w + h);
            // Mid-frequency sinusoid + bandlimited noise.
            let s = (((i as f32 * 0.32 + j as f32 * 0.21).sin() * 28.0) as i32).clamp(-40, 40);
            let n = (next_byte() - 128) / 24;
            let v = (g as i32 + s + n).clamp(0, 255);
            y[j * w + i] = v as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-50, 50)) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-50, 50)) as u8;
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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
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

fn make_params(width: u32, height: u32, bit_rate: Option<u64>) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.media_type = MediaType::Video;
    p.width = Some(width);
    p.height = Some(height);
    p.pixel_format = Some(PixelFormat::Yuv422P);
    p.bit_rate = bit_rate;
    p
}

fn encode_with_qi(width: u32, height: u32, qi: Option<u8>, src: &VideoFrame) -> Vec<u8> {
    let params = make_params(width, height, None);
    let cfg = match qi {
        Some(qi) => EncoderConfig::default().with_quantization_index(qi),
        None => EncoderConfig::default(),
    };
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(src.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    pkt.data
}

fn round_trip(packet_bytes: &[u8], width: u32, height: u32) -> VideoFrame {
    let params = make_params(width, height, None);
    let mut reg = CodecRegistry::new();
    oxideav_prores::register(&mut reg);
    let mut dec = reg.make_decoder(&params).expect("make_decoder");
    let mut pkt = oxideav_core::Packet::new(
        0,
        oxideav_core::TimeBase::new(1, 30),
        packet_bytes.to_vec(),
    );
    pkt.flags.keyframe = true;
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    }
}

#[test]
fn quant_index_default_matches_profile_default() {
    // Standard profile default qi = 4. EncoderConfig with qi=None must
    // produce a byte-identical packet to EncoderConfig with qi=Some(4).
    let width = 64u32;
    let height = 48u32;
    let src = synth_textured(width, height);
    let pkt_default = encode_with_qi(width, height, None, &src);
    let pkt_explicit = encode_with_qi(width, height, Some(4), &src);
    assert_eq!(
        pkt_default, pkt_explicit,
        "EncoderConfig with quantization_index=None must match \
         EncoderConfig with quantization_index=Some(profile_default) byte-for-byte"
    );
}

#[test]
fn quant_index_lower_grows_packet() {
    // Across qi=2 (highest quality) → qi=12 (lowest quality), packet
    // size must be monotonically non-increasing — and strictly
    // decreasing on a textured input.
    let width = 96u32;
    let height = 64u32;
    let src = synth_textured(width, height);
    let mut prev_size = usize::MAX;
    for qi in [2u8, 4, 6, 8, 10, 12] {
        let bytes = encode_with_qi(width, height, Some(qi), &src);
        eprintln!("qi={qi:>2}: packet={} bytes", bytes.len());
        assert!(
            bytes.len() < prev_size,
            "qi={qi} packet ({} B) must be strictly smaller than the previous qi packet ({} B)",
            bytes.len(),
            prev_size,
        );
        prev_size = bytes.len();
    }
}

#[test]
fn quant_index_lower_improves_psnr() {
    // qi=2 must round-trip with strictly higher luma PSNR than qi=12.
    let width = 96u32;
    let height = 64u32;
    let src = synth_textured(width, height);

    let pkt_lo = encode_with_qi(width, height, Some(2), &src);
    let pkt_hi = encode_with_qi(width, height, Some(12), &src);
    let dec_lo = round_trip(&pkt_lo, width, height);
    let dec_hi = round_trip(&pkt_hi, width, height);
    let p_lo = psnr(&src.planes[0].data, &dec_lo.planes[0].data);
    let p_hi = psnr(&src.planes[0].data, &dec_hi.planes[0].data);
    eprintln!("qi=2  luma PSNR = {p_lo:.2} dB ({} B packet)", pkt_lo.len());
    eprintln!("qi=12 luma PSNR = {p_hi:.2} dB ({} B packet)", pkt_hi.len());
    assert!(
        p_lo > p_hi + 3.0,
        "qi=2 PSNR ({p_lo:.2} dB) must beat qi=12 PSNR ({p_hi:.2} dB) by > 3 dB"
    );
}

#[test]
fn quant_index_overrides_profile_default_on_proxy() {
    // bit_rate hint <= 70_000_000 selects Proxy (default qi=8). Pin the
    // quant_index to 2 via the config and verify the bitstream is the
    // higher-quality one.
    let width = 64u32;
    let height = 48u32;
    let src = synth_textured(width, height);
    let params = make_params(width, height, Some(50_000_000)); // → Proxy

    let mut enc_default =
        make_encoder_with_config(&params, EncoderConfig::default()).expect("make_encoder default");
    enc_default
        .send_frame(&Frame::Video(src.clone()))
        .expect("send_frame default");
    let pkt_default = enc_default
        .receive_packet()
        .expect("receive_packet default");

    let mut enc_override = make_encoder_with_config(
        &params,
        EncoderConfig::default().with_quantization_index(2),
    )
    .expect("make_encoder override");
    enc_override
        .send_frame(&Frame::Video(src.clone()))
        .expect("send_frame override");
    let pkt_override = enc_override
        .receive_packet()
        .expect("receive_packet override");

    assert!(
        pkt_override.data.len() > pkt_default.data.len(),
        "qi=2 override Proxy packet ({} B) must be larger than the default qi=8 Proxy packet \
         ({} B) — finer qscale = more bits",
        pkt_override.data.len(),
        pkt_default.data.len(),
    );

    // And both must roundtrip cleanly through the registry decoder.
    let dec_default = round_trip(&pkt_default.data, width, height);
    let dec_override = round_trip(&pkt_override.data, width, height);
    let p_default = psnr(&src.planes[0].data, &dec_default.planes[0].data);
    let p_override = psnr(&src.planes[0].data, &dec_override.planes[0].data);
    assert!(
        p_override > p_default,
        "qi=2 override PSNR ({p_override:.2} dB) must beat qi=8 default PSNR ({p_default:.2} dB)",
    );
}

#[test]
fn quant_index_zero_rejected() {
    let params = make_params(64, 48, None);
    let res =
        make_encoder_with_config(&params, EncoderConfig::default().with_quantization_index(0));
    assert!(
        res.is_err(),
        "EncoderConfig with quantization_index=0 must be rejected (RDD 36 §7.3 range is 1..=224)"
    );
}

#[test]
fn quant_index_225_rejected() {
    let params = make_params(64, 48, None);
    let res =
        make_encoder_with_config(&params, EncoderConfig::default().with_quantization_index(225));
    assert!(
        res.is_err(),
        "EncoderConfig with quantization_index=225 must be rejected (RDD 36 §7.3 range is 1..=224)"
    );
}

#[test]
fn quant_index_boundaries_accepted() {
    // qi=1 (finest) and qi=224 (coarsest) are in-range and must encode
    // a complete frame.
    let width = 64u32;
    let height = 48u32;
    let src = synth_textured(width, height);
    for qi in [1u8, 224u8] {
        let bytes = encode_with_qi(width, height, Some(qi), &src);
        assert!(
            bytes.len() > 8,
            "qi={qi} must produce a non-trivial packet (got {} bytes)",
            bytes.len()
        );
        // Decode must succeed even at the coarsest valid index.
        let _ = round_trip(&bytes, width, height);
    }
}
