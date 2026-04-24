// Parallel-array index loops are idiomatic in codec/DCT code; skip the lint.
#![allow(clippy::needless_range_loop)]

//! Apple ProRes codec — minimal pure-Rust decoder + encoder.
//!
//! Scope: all six ProRes video profiles, selected via
//! `CodecParameters::bit_rate` and `pixel_format`:
//!
//! * 422 Proxy / LT / Standard / HQ for 4:2:2 Y'CbCr 8-bit
//!   (`PixelFormat::Yuv422P`, fourccs `apco`/`apcs`/`apcn`/`apch`).
//! * 4444 + 4444 XQ for 4:4:4 Y'CbCr 8-bit (`PixelFormat::Yuv444P`,
//!   fourccs `ap4h`/`ap4x`). The alpha plane of RDD 36 Annex B is
//!   not carried — the core pixel-format enum does not yet include
//!   `Yuva444P`, so the A' coding layer is skipped.
//!
//! The wire format follows SMPTE RDD 36 structurally (frame header
//! with `icpf` magic, picture header, per-slice table, slices holding
//! 8x8 DCT coefficients for 4 luma + 2 Cb + 2 Cr blocks per MB in
//! 4:2:2, or 4 Y + 4 Cb + 4 Cr blocks per MB in 4:4:4) but uses a
//! simplified entropy layer (unsigned / signed exp-Golomb on
//! zig-zag-scanned coefficients) that is internally round-trip-exact
//! but **not** bit-compatible with Apple ProRes in the wild. See the
//! module docs of [`bitstream`] and [`slice`] for the deviation.
//!
//! ### Module layout
//!
//! * [`bitstream`] — MSB-first bit reader/writer, unsigned/signed exp-Golomb.
//! * [`dct`]       — Textbook f32 8x8 forward/inverse DCT.
//! * [`quant`]     — Hard-coded luma/chroma quant matrices + zig-zag table.
//! * [`slice`]     — Per-slice pack/unpack of coefficient blocks.
//! * [`frame`]     — Frame + picture header layouts.
//! * [`decoder`]   — `Packet -> VideoFrame` (Yuv422P / Yuv444P).
//! * [`encoder`]   — `VideoFrame` (Yuv422P / Yuv444P) -> `Packet`.
//!
//! ### Container dispatch
//!
//! MP4 / MOV demuxers can recognise a ProRes stream in two ways:
//!
//! 1. **Dynamic** — call [`register`] on a `CodecRegistry`; the crate
//!    claims the six profile FourCCs as case-insensitive tags so
//!    `CodecRegistry::resolve_tag` returns `CodecId::new("prores")`
//!    for any of them. This is what `oxideav-mp4`'s demuxer does when
//!    a resolver is plumbed in.
//! 2. **Static** — call [`codec_id_for_fourcc`] for the same mapping
//!    without touching a registry. Useful for demuxers that cannot
//!    consult a resolver (early probing, unit tests). [`PRORES_FOURCCS`]
//!    enumerates the six canonical lower-case tags.

pub mod bitstream;
pub mod dct;
pub mod decoder;
pub mod encoder;
pub mod frame;
pub mod quant;
pub mod slice;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag, PixelFormat};

/// Public codec id.
pub const CODEC_ID_STR: &str = "prores";

/// All six MP4 / MOV `VisualSampleEntry` FourCCs that identify a
/// ProRes bitstream. Canonical lower-case spelling as defined by
/// Apple's ProRes white paper (April 2022):
///
/// | fourcc | profile                                     |
/// |--------|---------------------------------------------|
/// | apco   | Apple ProRes 422 Proxy                      |
/// | apcs   | Apple ProRes 422 LT                         |
/// | apcn   | Apple ProRes 422 (Standard)                 |
/// | apch   | Apple ProRes 422 HQ                         |
/// | ap4h   | Apple ProRes 4444                           |
/// | ap4x   | Apple ProRes 4444 XQ                        |
///
/// Container dispatch is case-insensitive in practice (demuxers
/// upper-case for comparison, some `.mov`s stored them as `APCN`
/// etc. since QuickTime originally normalised FourCCs), so callers
/// should compare case-insensitively. This constant gives the
/// canonical lower-case spelling only.
pub const PRORES_FOURCCS: [&[u8; 4]; 6] = [b"apco", b"apcs", b"apcn", b"apch", b"ap4h", b"ap4x"];

/// Returns `Some(CodecId::new("prores"))` if `fourcc` (case-insensitive)
/// is one of the six ProRes MP4/MOV `VisualSampleEntry` FourCCs,
/// otherwise `None`. Intended as a static fallback for demuxers that
/// cannot consult a `CodecRegistry` — callers that _do_ have a
/// registry should just register this crate via [`register`] and
/// consult `CodecRegistry::resolve_tag`, which already claims the
/// same six FourCCs as case-insensitive tags.
pub fn codec_id_for_fourcc(fourcc: &[u8; 4]) -> Option<CodecId> {
    let mut upper = [0u8; 4];
    for i in 0..4 {
        upper[i] = fourcc[i].to_ascii_uppercase();
    }
    match &upper {
        b"APCO" | b"APCS" | b"APCN" | b"APCH" | b"AP4H" | b"AP4X" => {
            Some(CodecId::new(CODEC_ID_STR))
        }
        _ => None,
    }
}

/// Returns the matching [`frame::Profile`] for a given MP4/MOV FourCC
/// (case-insensitive), or `None` for non-ProRes tags.
pub fn profile_for_fourcc(fourcc: &[u8; 4]) -> Option<frame::Profile> {
    let mut upper = [0u8; 4];
    for i in 0..4 {
        upper[i] = fourcc[i].to_ascii_uppercase();
    }
    Some(match &upper {
        b"APCO" => frame::Profile::Proxy,
        b"APCS" => frame::Profile::Lt,
        b"APCN" => frame::Profile::Standard,
        b"APCH" => frame::Profile::Hq,
        b"AP4H" => frame::Profile::Prores4444,
        b"AP4X" => frame::Profile::Prores4444Xq,
        _ => return None,
    })
}

/// Register the ProRes decoder + encoder for all six profiles
/// (422 Proxy/LT/Standard/HQ and 4444 / 4444 XQ).
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("prores_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_pixel_format(PixelFormat::Yuv422P)
        .with_pixel_format(PixelFormat::Yuv444P);
    // FourCC claims — ProRes profile FourCCs. Each profile gets its own
    // FourCC at the container level but they all decode through the
    // same path here. Unambiguous.
    //   APCO = 422 Proxy, APCS = 422 LT, APCN = 422 (Standard),
    //   APCH = 422 HQ, AP4H = 4444, AP4X = 4444 XQ.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tags([
                CodecTag::fourcc(b"APCO"),
                CodecTag::fourcc(b"APCS"),
                CodecTag::fourcc(b"APCN"),
                CodecTag::fourcc(b"APCH"),
                CodecTag::fourcc(b"AP4H"),
                CodecTag::fourcc(b"AP4X"),
            ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{
        CodecId, CodecParameters, Frame, MediaType, PixelFormat, TimeBase, VideoFrame,
    };

    /// Build a 64x48 gradient in Yuv422P. Values are deliberately
    /// smooth so the codec's lossy path can hit reasonable PSNR at
    /// `quant_index = 4`.
    fn synthetic_gradient(width: u32, height: u32) -> VideoFrame {
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
        }
        for j in 0..h {
            for i in 0..cw {
                cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-64, 64)) as u8;
                cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-64, 64)) as u8;
            }
        }
        VideoFrame {
            format: PixelFormat::Yuv422P,
            width,
            height,
            pts: Some(0),
            time_base: TimeBase::new(1, 30),
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

    fn psnr(orig: &[u8], decoded: &[u8]) -> f64 {
        assert_eq!(orig.len(), decoded.len());
        let mut mse = 0.0f64;
        for (a, b) in orig.iter().zip(decoded.iter()) {
            let d = *a as f64 - *b as f64;
            mse += d * d;
        }
        mse /= orig.len() as f64;
        if mse == 0.0 {
            return 120.0;
        }
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    #[test]
    fn encoder_decoder_roundtrip_psnr() {
        // 64x48 fits exactly 4x3 MBs of 16x16 — no padding edge cases.
        let width = 64u32;
        let height = 48u32;
        let original = synthetic_gradient(width, height);

        let mut enc_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        enc_params.media_type = MediaType::Video;
        enc_params.width = Some(width);
        enc_params.height = Some(height);
        enc_params.pixel_format = Some(PixelFormat::Yuv422P);

        let mut reg = oxideav_codec::CodecRegistry::new();
        register(&mut reg);
        let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
        encoder
            .send_frame(&Frame::Video(original.clone()))
            .expect("send_frame");
        let pkt = encoder.receive_packet().expect("receive_packet");

        let dec_params = enc_params.clone();
        let mut decoder = reg.make_decoder(&dec_params).expect("make_decoder");
        decoder.send_packet(&pkt).expect("send_packet");
        let frame = decoder.receive_frame().expect("receive_frame");
        let decoded = match frame {
            Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };

        assert_eq!(decoded.format, PixelFormat::Yuv422P);
        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.planes.len(), 3);

        for (i, (o, d)) in original
            .planes
            .iter()
            .zip(decoded.planes.iter())
            .enumerate()
        {
            assert_eq!(o.data.len(), d.data.len(), "plane {i} size mismatch");
            let p = psnr(&o.data, &d.data);
            assert!(p > 30.0, "plane {i} PSNR too low: {p:.2} dB (want > 30)");
            eprintln!("plane {i} PSNR = {p:.2} dB");
        }
    }

    #[test]
    fn decoder_registered() {
        let mut reg = oxideav_codec::CodecRegistry::new();
        register(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(reg.has_encoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn codec_id_for_fourcc_maps_all_six() {
        for fc in PRORES_FOURCCS {
            assert_eq!(codec_id_for_fourcc(fc), Some(CodecId::new(CODEC_ID_STR)));
        }
    }

    #[test]
    fn codec_id_for_fourcc_is_case_insensitive() {
        assert_eq!(
            codec_id_for_fourcc(b"APCH"),
            Some(CodecId::new(CODEC_ID_STR))
        );
        assert_eq!(
            codec_id_for_fourcc(b"apch"),
            Some(CodecId::new(CODEC_ID_STR))
        );
        assert_eq!(
            codec_id_for_fourcc(b"ApCh"),
            Some(CodecId::new(CODEC_ID_STR))
        );
    }

    #[test]
    fn codec_id_for_fourcc_rejects_non_prores() {
        assert_eq!(codec_id_for_fourcc(b"avc1"), None);
        assert_eq!(codec_id_for_fourcc(b"hvc1"), None);
        assert_eq!(codec_id_for_fourcc(b"mp4v"), None);
        // Non-ProRes `a*` tags: close but not ours.
        assert_eq!(codec_id_for_fourcc(b"alac"), None);
        assert_eq!(codec_id_for_fourcc(b"av01"), None);
    }

    #[test]
    fn profile_for_fourcc_roundtrips_via_profile_fourcc() {
        for p in [
            frame::Profile::Proxy,
            frame::Profile::Lt,
            frame::Profile::Standard,
            frame::Profile::Hq,
            frame::Profile::Prores4444,
            frame::Profile::Prores4444Xq,
        ] {
            let fc = p.fourcc();
            assert_eq!(profile_for_fourcc(fc), Some(p), "fourcc roundtrip");
            // Upper-case variant must also map.
            let mut up = *fc;
            up.make_ascii_uppercase();
            assert_eq!(profile_for_fourcc(&up), Some(p));
        }
        assert_eq!(profile_for_fourcc(b"mp4v"), None);
    }

    #[test]
    fn registry_recognizes_prores_fourcc_tags() {
        use oxideav_core::stream::{CodecResolver, ProbeContext};
        use oxideav_core::CodecTag;
        let mut reg = oxideav_codec::CodecRegistry::new();
        register(&mut reg);
        for fc in PRORES_FOURCCS {
            let tag = CodecTag::fourcc(fc);
            let ctx = ProbeContext::new(&tag);
            let id = reg.resolve_tag(&ctx).expect("resolve_tag");
            assert_eq!(id, CodecId::new(CODEC_ID_STR), "fourcc {fc:?}");
        }
    }

    /// Build a 64x48 gradient in Yuv444P (chroma at full luma resolution).
    fn synthetic_gradient_444(width: u32, height: u32) -> VideoFrame {
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
            format: PixelFormat::Yuv444P,
            width,
            height,
            pts: Some(0),
            time_base: TimeBase::new(1, 30),
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

    #[test]
    fn encoder_decoder_roundtrip_psnr_4444() {
        let width = 64u32;
        let height = 48u32;
        let original = synthetic_gradient_444(width, height);

        let mut enc_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        enc_params.media_type = MediaType::Video;
        enc_params.width = Some(width);
        enc_params.height = Some(height);
        enc_params.pixel_format = Some(PixelFormat::Yuv444P);

        let mut reg = oxideav_codec::CodecRegistry::new();
        register(&mut reg);
        let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
        encoder
            .send_frame(&Frame::Video(original.clone()))
            .expect("send_frame");
        let pkt = encoder.receive_packet().expect("receive_packet");

        let dec_params = enc_params.clone();
        let mut decoder = reg.make_decoder(&dec_params).expect("make_decoder");
        decoder.send_packet(&pkt).expect("send_packet");
        let frame = decoder.receive_frame().expect("receive_frame");
        let decoded = match frame {
            Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };

        assert_eq!(decoded.format, PixelFormat::Yuv444P);
        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.planes.len(), 3);

        for (i, (o, d)) in original
            .planes
            .iter()
            .zip(decoded.planes.iter())
            .enumerate()
        {
            assert_eq!(o.data.len(), d.data.len(), "plane {i} size mismatch");
            let p = psnr(&o.data, &d.data);
            assert!(
                p > 30.0,
                "4444 plane {i} PSNR too low: {p:.2} dB (want > 30)"
            );
            eprintln!("4444 plane {i} PSNR = {p:.2} dB");
        }
    }
}
