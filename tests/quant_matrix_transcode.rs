//! Transcode-forwarding contract for quantisation matrices
//! (`QuantMatrices::from_header` → `EncoderConfig::with_quant_matrices`).
//!
//! A transcode wants to preserve the source stream's exact quantisation
//! matrices. This suite drives the full path for every corpus profile:
//! decode a reference `input.mov` frame header, recover its carried
//! matrix pair via `from_header`, feed it back into the encoder, and
//! confirm the re-encoded stream carries the **identical** matrices and
//! decodes cleanly. Unlike `quant_matrix_signature.rs` (which pins the
//! crate's own hard-coded presets), this proves the encoder faithfully
//! re-emits an *arbitrary* matrix pair it did not itself generate —
//! including Proxy's distinct chroma table. Validator-independent: the
//! reference stream is opaque data; only this crate's parser + encoder +
//! decoder run.

use std::fs;
use std::path::PathBuf;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile};
use oxideav_prores::quant::QuantMatrices;

const CODEC_ID_STR: &str = "prores";

fn fixture_frame(name: &str) -> Vec<u8> {
    let path = PathBuf::from("../../docs/video/prores/fixtures")
        .join(name)
        .join("input.mov");
    let container = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let needle = b"icpf";
    let mut i = 4usize;
    while i + 4 <= container.len() {
        if &container[i..i + 4] == needle {
            let size_off = i - 4;
            let frame_size =
                u32::from_be_bytes(container[size_off..size_off + 4].try_into().unwrap()) as usize;
            let end = size_off + frame_size;
            if end <= container.len() && frame_size >= 8 {
                return container[size_off..end].to_vec();
            }
        }
        i += 1;
    }
    panic!("no ProRes frame found in {name}");
}

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

/// Decode a fixture header, recover its matrices via `from_header`,
/// re-encode a synthesised frame carrying those matrices, and confirm
/// the re-encoded header preserves the source matrix pair exactly.
fn transcode_matrices(name: &str, profile: Profile) {
    let src_frame = fixture_frame(name);
    let (src_fh, _) = parse_frame(&src_frame).expect("parse source header");
    let forwarded = QuantMatrices::from_header(&src_fh);

    let chroma = profile.chroma_format();
    assert_eq!(chroma, src_fh.chroma_format, "{name}: chroma format");

    let (width, height) = (96u32, 64u32);
    let src = synth(width, height, chroma);
    let cfg = EncoderConfig::for_profile(profile).with_quant_matrices(forwarded);
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pixel_format(chroma));
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(src.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet").data;

    // The re-encoded header preserves the source's matrices exactly.
    let (out_fh, _) = parse_frame(&pkt).expect("parse re-encoded header");
    assert_eq!(
        out_fh.luma_qmat, src_fh.luma_qmat,
        "{name}: forwarded luma matrix"
    );
    assert_eq!(
        out_fh.chroma_qmat, src_fh.chroma_qmat,
        "{name}: forwarded chroma matrix"
    );

    // And the re-encoded stream decodes.
    let out = decode(&pkt, width, height, chroma);
    assert_eq!(out.planes[0].data.len(), (width * height) as usize);
}

#[test]
fn transcode_proxy_forwards_distinct_chroma() {
    // Proxy is the important case: its chroma table differs from luma,
    // so a faithful transcode must forward both distinct tables.
    let (fh, _) = parse_frame(&fixture_frame("proxy-1280x720")).unwrap();
    assert_ne!(fh.luma_qmat, fh.chroma_qmat, "proxy source chroma differs");
    transcode_matrices("proxy-1280x720", Profile::Proxy);
}

#[test]
fn transcode_lt() {
    transcode_matrices("lt-1280x720", Profile::Lt);
}

#[test]
fn transcode_standard() {
    transcode_matrices("sq-1920x1080", Profile::Standard);
}

#[test]
fn transcode_hq() {
    transcode_matrices("hq-1920x1080", Profile::Hq);
}

#[test]
fn transcode_4444() {
    transcode_matrices("4444-1920x1080", Profile::Prores4444);
}

#[test]
fn transcode_4444xq() {
    transcode_matrices("4444xq-1920x1080", Profile::Prores4444Xq);
}
