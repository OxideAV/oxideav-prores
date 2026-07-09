//! Encoder-side coverage of the per-profile signature quantisation
//! preset (`EncoderConfig::signature_for_profile`).
//!
//! For every profile, encoding a synthesised frame with the signature
//! preset must (1) carry the profile's native quantisation matrix pair
//! in the emitted frame header, (2) use the minimal RDD 36 §6.1.1
//! carriage — both tables for Proxy, a single luma table (chroma via the
//! §6.1.1 fallback) for the other five — and (3) self-round-trip through
//! the decoder at a healthy PSNR. Validator-independent: everything is
//! this crate's own encoder + decoder.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile};
use oxideav_prores::quant::QuantMatrices;

const CODEC_ID_STR: &str = "prores";

fn synth(width: u32, height: u32, chroma: ChromaFormat) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = match chroma {
        ChromaFormat::Y422 => w / 2,
        ChromaFormat::Y444 => w,
    };
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            let g = (i + j) * 200 / (w + h);
            let s = (((i as f32 * 0.30).sin() + (j as f32 * 0.25).cos()) * 24.0) as i32;
            y[j * w + i] = (g as i32 + s).clamp(0, 255) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2).clamp(-40, 40))) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2).clamp(-40, 40))) as u8;
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

fn pixel_format(chroma: ChromaFormat) -> PixelFormat {
    match chroma {
        ChromaFormat::Y422 => PixelFormat::Yuv422P,
        ChromaFormat::Y444 => PixelFormat::Yuv444P,
    }
}

fn encode(profile: Profile, width: u32, height: u32, src: &VideoFrame) -> Vec<u8> {
    let chroma = profile.chroma_format();
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pixel_format(chroma));
    let cfg = EncoderConfig::signature_for_profile(profile);
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(src.clone()))
        .expect("send_frame");
    enc.receive_packet().expect("receive_packet").data
}

fn decode(packet: &[u8], width: u32, height: u32, chroma: ChromaFormat) -> VideoFrame {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pixel_format(chroma));
    let mut reg = CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make_decoder");
    let mut pkt = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 30), packet.to_vec());
    pkt.flags.keyframe = true;
    dec.send_packet(&pkt).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
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

fn check_profile(profile: Profile) {
    let (width, height) = (96u32, 64u32);
    let chroma = profile.chroma_format();
    let src = synth(width, height, chroma);

    let pkt = encode(profile, width, height, &src);
    let (fh, _) = parse_frame(&pkt).expect("parse header");

    // The emitted header carries the profile's native signature pair.
    let sig = QuantMatrices::signature_for_profile(profile);
    assert_eq!(fh.luma_qmat, sig.luma, "{profile:?}: emitted luma matrix");
    assert_eq!(
        fh.chroma_qmat, sig.chroma,
        "{profile:?}: emitted chroma matrix"
    );

    // Minimal carriage: both tables for Proxy, one for the rest.
    let (load_l, load_c) = sig.wire_flags();
    assert_eq!(
        (
            fh.load_luma_quantization_matrix,
            fh.load_chroma_quantization_matrix
        ),
        (load_l, load_c),
        "{profile:?}: carriage flags"
    );
    let expected_header = 20 + if load_l { 64 } else { 0 } + if load_c { 64 } else { 0 };
    assert_eq!(
        fh.frame_header_size as usize, expected_header,
        "{profile:?}: frame_header_size"
    );

    // Self-round-trips through the decoder at a healthy luma PSNR.
    let out = decode(&pkt, width, height, chroma);
    let luma = psnr(&src.planes[0].data, &out.planes[0].data);
    assert!(luma > 30.0, "{profile:?}: luma PSNR {luma:.1} dB too low");
}

#[test]
fn signature_encode_proxy() {
    check_profile(Profile::Proxy);
    // Proxy is the one profile that writes both quant tables.
    let src = synth(96, 64, ChromaFormat::Y422);
    let pkt = encode(Profile::Proxy, 96, 64, &src);
    let (fh, _) = parse_frame(&pkt).unwrap();
    assert!(fh.load_luma_quantization_matrix && fh.load_chroma_quantization_matrix);
    assert_ne!(fh.luma_qmat, fh.chroma_qmat);
}

#[test]
fn signature_encode_lt() {
    check_profile(Profile::Lt);
}

#[test]
fn signature_encode_standard() {
    check_profile(Profile::Standard);
}

#[test]
fn signature_encode_hq() {
    check_profile(Profile::Hq);
}

#[test]
fn signature_encode_4444() {
    check_profile(Profile::Prores4444);
}

#[test]
fn signature_encode_4444xq() {
    check_profile(Profile::Prores4444Xq);
}
