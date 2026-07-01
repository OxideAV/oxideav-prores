//! RDD 36 §6.1.1 quantization-matrix carriage — all four
//! (`load_luma_quantization_matrix`, `load_chroma_quantization_matrix`)
//! combinations, end to end.
//!
//! The corpus fixtures are uniformly flags `(1, 1)` (both custom tables
//! present, even when chroma == luma). This suite drives our own encoder
//! across every legal §6.1.1 carriage and pins, for each, the wire flags,
//! the derived [`QuantizationMatrixSource`], the frame-header size, the
//! reconstructed luma/chroma matrices (§6.1.1 fallback), and a clean
//! registry decode. Validator-independent.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::encode_frame_with_qmats;
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile, QuantizationMatrixSource};
use oxideav_prores::quant::{
    QuantMatrices, DEFAULT_QMAT, PERCEPTUAL_CHROMA_QMAT, PERCEPTUAL_LUMA_QMAT,
};
use oxideav_prores::CODEC_ID_STR;

const W: u32 = 96;
const H: u32 = 64;

/// A small 4:2:2 gradient frame with mild texture.
fn synth_422() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (((i + j) * 200 / (w + h)) as u8).wrapping_add(((i * j) as u8) & 0x1F);
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

fn encode(qm: QuantMatrices) -> Vec<u8> {
    encode_frame_with_qmats(
        &synth_422(),
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        qm,
    )
    .expect("encode")
}

fn decode_ok(pkt: &[u8]) {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    let mut reg = CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make_decoder");
    let mut p = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 30), pkt.to_vec());
    p.flags.keyframe = true;
    dec.send_packet(&p).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => {
            assert_eq!(v.planes.len(), 3);
            assert_eq!(v.planes[0].data.len(), v.planes[0].stride * H as usize);
        }
        _ => panic!("expected video frame"),
    }
}

/// Case (0,0): both default. Header 20; source Default; both matrices
/// reconstruct to the §7.2 all-4s default.
#[test]
fn carriage_both_default() {
    let pkt = encode(QuantMatrices::flat());
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert!(!fh.load_luma_quantization_matrix);
    assert!(!fh.load_chroma_quantization_matrix);
    assert_eq!(
        fh.quantization_matrix_source(),
        QuantizationMatrixSource::Default
    );
    assert_eq!(fh.frame_header_size, 20);
    assert_eq!(fh.luma_qmat, DEFAULT_QMAT);
    assert_eq!(fh.chroma_qmat, DEFAULT_QMAT);
    decode_ok(&pkt);
}

/// Case (1,0): custom luma, chroma copies luma via the §6.1.1 fallback.
/// Header 84 (one table); source LumaCustom.
#[test]
fn carriage_luma_custom_chroma_copies() {
    let qm = QuantMatrices {
        luma: PERCEPTUAL_LUMA_QMAT,
        chroma: PERCEPTUAL_LUMA_QMAT,
    };
    let pkt = encode(qm);
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert!(fh.load_luma_quantization_matrix);
    assert!(!fh.load_chroma_quantization_matrix);
    assert_eq!(
        fh.quantization_matrix_source(),
        QuantizationMatrixSource::LumaCustom
    );
    assert_eq!(fh.frame_header_size, 84);
    assert_eq!(fh.luma_qmat, PERCEPTUAL_LUMA_QMAT);
    // §6.1.1 fallback: chroma reconstructs to the (custom) luma matrix.
    assert_eq!(fh.chroma_qmat, PERCEPTUAL_LUMA_QMAT);
    decode_ok(&pkt);
}

/// Case (0,1): default luma, custom chroma — the form the previous
/// encoder derivation could not emit. Header 84 (one table); source
/// CustomChroma; luma reconstructs to the default.
#[test]
fn carriage_default_luma_custom_chroma() {
    let qm = QuantMatrices {
        luma: DEFAULT_QMAT,
        chroma: PERCEPTUAL_CHROMA_QMAT,
    };
    let pkt = encode(qm);
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert!(!fh.load_luma_quantization_matrix);
    assert!(fh.load_chroma_quantization_matrix);
    assert_eq!(
        fh.quantization_matrix_source(),
        QuantizationMatrixSource::CustomChroma
    );
    assert_eq!(fh.frame_header_size, 84);
    assert_eq!(fh.luma_qmat, DEFAULT_QMAT);
    assert_eq!(fh.chroma_qmat, PERCEPTUAL_CHROMA_QMAT);
    decode_ok(&pkt);
}

/// Case (1,1): both custom and distinct. Header 148 (two tables); source
/// CustomChroma.
#[test]
fn carriage_both_custom() {
    let qm = QuantMatrices {
        luma: PERCEPTUAL_LUMA_QMAT,
        chroma: PERCEPTUAL_CHROMA_QMAT,
    };
    let pkt = encode(qm);
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert!(fh.load_luma_quantization_matrix);
    assert!(fh.load_chroma_quantization_matrix);
    assert_eq!(
        fh.quantization_matrix_source(),
        QuantizationMatrixSource::CustomChroma
    );
    assert_eq!(fh.frame_header_size, 148);
    assert_eq!(fh.luma_qmat, PERCEPTUAL_LUMA_QMAT);
    assert_eq!(fh.chroma_qmat, PERCEPTUAL_CHROMA_QMAT);
    decode_ok(&pkt);
}

/// Custom-luma + default-chroma: chroma differs from luma, so it must be
/// carried explicitly — this is (1,1) with an all-4s chroma table, NOT
/// (1,0). Without carrying it the §6.1.1 fallback would wrongly copy the
/// custom luma into chroma.
#[test]
fn carriage_custom_luma_default_chroma_needs_both() {
    let qm = QuantMatrices {
        luma: PERCEPTUAL_LUMA_QMAT,
        chroma: DEFAULT_QMAT,
    };
    let pkt = encode(qm);
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert!(fh.load_luma_quantization_matrix);
    assert!(fh.load_chroma_quantization_matrix);
    assert_eq!(fh.frame_header_size, 148);
    assert_eq!(fh.luma_qmat, PERCEPTUAL_LUMA_QMAT);
    assert_eq!(fh.chroma_qmat, DEFAULT_QMAT);
    decode_ok(&pkt);
}

/// Header-size ordering across the carriage forms: default (20) < single
/// table (84) < both tables (148).
#[test]
fn carriage_header_size_ordering() {
    let h_default = {
        let (fh, _) = parse_frame(&encode(QuantMatrices::flat())).unwrap();
        fh.frame_header_size
    };
    let h_one = {
        let (fh, _) = parse_frame(&encode(QuantMatrices {
            luma: DEFAULT_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        }))
        .unwrap();
        fh.frame_header_size
    };
    let h_both = {
        let (fh, _) = parse_frame(&encode(QuantMatrices {
            luma: PERCEPTUAL_LUMA_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        }))
        .unwrap();
        fh.frame_header_size
    };
    assert!(
        h_default < h_one && h_one < h_both,
        "{h_default} {h_one} {h_both}"
    );
    assert_eq!((h_default, h_one, h_both), (20, 84, 148));
}
