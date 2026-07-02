//! RDD 36 §6.1.1 quantization-matrix carriage across the interlaced
//! two-picture path and the 4:4:4 chroma-doubling path.
//!
//! The carriage flags are written once per frame_header() and shared by
//! every picture() in the frame, so an interlaced frame's two field
//! pictures both dequantise with the reconstructed matrices, and a 4:4:4
//! frame applies the (possibly custom) chroma matrix to full-resolution
//! Cb/Cr. This suite drives the compact and full carriage forms through
//! both code paths via the registry `EncoderConfig` path and asserts the
//! wire flags plus a clean decode. Validator-independent.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{parse_frame, Profile};
use oxideav_prores::quant::{
    QuantMatrices, DEFAULT_QMAT, PERCEPTUAL_CHROMA_QMAT, PERCEPTUAL_LUMA_QMAT,
};
use oxideav_prores::CODEC_ID_STR;

const W: u32 = 64;
const H: u32 = 48;

fn plane(stride: usize, rows: usize, seed: u8) -> VideoPlane {
    let mut data = vec![0u8; stride * rows];
    for (idx, p) in data.iter_mut().enumerate() {
        *p = ((idx as u8).wrapping_mul(31)).wrapping_add(seed) ^ ((idx >> 4) as u8);
    }
    VideoPlane { stride, data }
}

fn synth_422() -> VideoFrame {
    let (w, h, cw) = (W as usize, H as usize, W as usize / 2);
    VideoFrame {
        pts: Some(0),
        planes: vec![plane(w, h, 16), plane(cw, h, 110), plane(cw, h, 140)],
    }
}

fn synth_444() -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    VideoFrame {
        pts: Some(0),
        planes: vec![plane(w, h, 16), plane(w, h, 110), plane(w, h, 140)],
    }
}

/// Encode `frame` through the registry `EncoderConfig` path and return the
/// packet plus its parsed (load_luma, load_chroma) wire flags.
fn encode(
    frame: &VideoFrame,
    pix: PixelFormat,
    profile: Profile,
    interlace_mode: u8,
    qm: QuantMatrices,
) -> (Vec<u8>, (bool, bool)) {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(pix);
    let cfg = EncoderConfig::default()
        .with_profile(profile)
        .with_interlace_mode(interlace_mode)
        .with_quant_matrices(qm);
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("receive");
    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.interlace_mode, interlace_mode);
    (
        pkt.data.clone(),
        (
            fh.load_luma_quantization_matrix,
            fh.load_chroma_quantization_matrix,
        ),
    )
}

fn decode_ok(pkt: &[u8], pix: PixelFormat, chroma_full: bool) {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(pix);
    let mut reg = CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make_decoder");
    let mut p = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 30), pkt.to_vec());
    p.flags.keyframe = true;
    dec.send_packet(&p).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => {
            assert_eq!(v.planes.len(), 3);
            // Luma spans the full frame height regardless of interlace.
            assert_eq!(v.planes[0].data.len(), v.planes[0].stride * H as usize);
            let expect_cw = if chroma_full {
                W as usize
            } else {
                W as usize / 2
            };
            assert!(v.planes[1].stride >= expect_cw);
        }
        _ => panic!("expected video frame"),
    }
}

// ---- interlaced 4:2:2 -------------------------------------------------

#[test]
fn interlaced_422_default_luma_custom_chroma() {
    let (pkt, flags) = encode(
        &synth_422(),
        PixelFormat::Yuv422P,
        Profile::Standard,
        1, // TFF
        QuantMatrices {
            luma: DEFAULT_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        },
    );
    assert_eq!(flags, (false, true));
    decode_ok(&pkt, PixelFormat::Yuv422P, false);
}

#[test]
fn interlaced_422_both_custom_bff() {
    let (pkt, flags) = encode(
        &synth_422(),
        PixelFormat::Yuv422P,
        Profile::Standard,
        2, // BFF
        QuantMatrices {
            luma: PERCEPTUAL_LUMA_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        },
    );
    assert_eq!(flags, (true, true));
    decode_ok(&pkt, PixelFormat::Yuv422P, false);
}

#[test]
fn interlaced_422_flat_stays_zero_flags() {
    let (pkt, flags) = encode(
        &synth_422(),
        PixelFormat::Yuv422P,
        Profile::Standard,
        1,
        QuantMatrices::flat(),
    );
    assert_eq!(flags, (false, false));
    decode_ok(&pkt, PixelFormat::Yuv422P, false);
}

// ---- progressive 4:4:4 ------------------------------------------------

#[test]
fn prores_444_default_luma_custom_chroma() {
    let (pkt, flags) = encode(
        &synth_444(),
        PixelFormat::Yuv444P,
        Profile::Prores4444,
        0,
        QuantMatrices {
            luma: DEFAULT_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        },
    );
    assert_eq!(flags, (false, true));
    decode_ok(&pkt, PixelFormat::Yuv444P, true);
}

#[test]
fn prores_444_both_custom() {
    let (pkt, flags) = encode(
        &synth_444(),
        PixelFormat::Yuv444P,
        Profile::Prores4444,
        0,
        QuantMatrices {
            luma: PERCEPTUAL_LUMA_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        },
    );
    assert_eq!(flags, (true, true));
    decode_ok(&pkt, PixelFormat::Yuv444P, true);
}

/// Interlaced 4:4:4 exercises both the two-picture split and the
/// full-resolution chroma path with a custom chroma matrix at once.
#[test]
fn interlaced_444_default_luma_custom_chroma() {
    let (pkt, flags) = encode(
        &synth_444(),
        PixelFormat::Yuv444P,
        Profile::Prores4444,
        1,
        QuantMatrices {
            luma: DEFAULT_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        },
    );
    assert_eq!(flags, (false, true));
    decode_ok(&pkt, PixelFormat::Yuv444P, true);
}
