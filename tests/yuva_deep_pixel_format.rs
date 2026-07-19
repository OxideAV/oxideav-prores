//! Deep alpha-typed pixel-format surface
//! (`PixelFormat::Yuva4(2|4)4P{10,12,16}Le`) on both codec directions,
//! plus the 16-bit colour surfaces (`Yuv4(2|4)4P16Le`) they build on.
//!
//! The 8-bit typed surface (`Yuva422P` / `Yuva444P`) made the 4-plane
//! layout a format contract but capped typed alpha at 8 bits; RDD 36
//! codes alpha at up to 16 bits (§6.1.1 Table 7 / §7.1.2 Table 14).
//! These tests pin the deep surfaces end-to-end:
//!
//! * **Decode** — a deep typed request guarantees a 4-plane frame:
//!   colour per §7.5.1 at the surface depth, coded alpha converted to
//!   the same depth per §7.5.2 on plane 3 (exact pass-through of
//!   16-bit coded alpha on the `*P16Le` surfaces — full RDD 36 alpha
//!   fidelity in the type system), and a synthesised fully-opaque
//!   plane on streams that code no alpha.
//! * **Encode** — a deep typed encoder reads 16-bit-word alpha input
//!   at the format's depth and codes 16-bit alpha
//!   (`alpha_channel_type = 2`) via the §7.5.2-mirror promotion,
//!   emitting wire bytes identical to the established free-function
//!   path fed the equivalent pre-promoted 16-bit alpha plane.
//! * **Contract violations** — wrong plane count, wrong alpha plane
//!   layout, contradictory explicit `AlphaChannelType` config, and
//!   chroma mismatches all surface clean `Err`s.
//!
//! Alpha is coded losslessly per §7.1.2, so alpha assertions are exact
//! (byte- or formula-exact), never a PSNR bar.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, VideoFrame};
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::{decode_packet_with_format, make_decoder, BitDepth, OutputRange};
use oxideav_prores::encoder::{
    encode_frame_with_alpha, make_encoder, make_encoder_with_config, EncoderConfig,
};
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

fn le_plane(samples: &[u16], samples_per_row: usize) -> VideoPlane {
    let mut data = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    VideoPlane {
        stride: samples_per_row * 2,
        data,
    }
}

/// Deep YCbCr content at `depth` (10/12/16-bit LE words), 4:4:4 or
/// 4:2:2 by `chroma`. Values stay inside a broadcast-legal-ish window
/// scaled to the depth so the lossy colour path round-trips at high
/// PSNR.
fn deep_colour_planes(
    w: usize,
    h: usize,
    depth: BitDepth,
    chroma: ChromaFormat,
) -> Vec<VideoPlane> {
    let max = depth.max_value();
    let cw = match chroma {
        ChromaFormat::Y422 => w / 2,
        ChromaFormat::Y444 => w,
    };
    let mut y = vec![0u16; w * h];
    let mut cb = vec![0u16; cw * h];
    let mut cr = vec![0u16; cw * h];
    for j in 0..h {
        for i in 0..w {
            let ramp = ((i * 3 + j * 2) % 180) as u32;
            y[j * w + i] = ((max / 8 + ramp * max / 256).min(max)) as u16;
        }
        for i in 0..cw {
            let dev = ((i as i32 - cw as i32 / 2) * (max as i32 / 512))
                .clamp(-(max as i32) / 8, max as i32 / 8);
            cb[j * cw + i] = ((max as i32 / 2 + dev).max(0) as u32).min(max) as u16;
            let dev_r = ((j as i32 - h as i32 / 2) * (max as i32 / 512))
                .clamp(-(max as i32) / 8, max as i32 / 8);
            cr[j * cw + i] = ((max as i32 / 2 + dev_r).max(0) as u32).min(max) as u16;
        }
    }
    vec![le_plane(&y, w), le_plane(&cb, cw), le_plane(&cr, cw)]
}

/// Full-resolution alpha plane at `depth`: a diagonal gradient sweeping
/// the entire representable range (both extreme codes included), so
/// the Table 12 run codes and the Table 14 difference codes are
/// exercised.
fn deep_alpha_samples(w: usize, h: usize, depth: BitDepth) -> Vec<u16> {
    let max = depth.max_value() as usize;
    let mut a = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            a[j * w + i] = (((i + j) * max) / (w + h - 2)) as u16;
        }
    }
    a
}

/// §7.5.2 conversion oracle (round-half-up integer form):
/// `round(out_max * alpha / in_max)`.
fn convert_752(alpha: u32, in_max: u32, out_max: u32) -> u16 {
    ((out_max as u64 * alpha as u64 + in_max as u64 / 2) / in_max as u64) as u16
}

fn read_u16s(plane: &VideoPlane) -> Vec<u16> {
    plane
        .data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn frame_with(planes: Vec<VideoPlane>) -> Frame {
    Frame::Video(VideoFrame {
        pts: Some(0),
        planes,
    })
}

/// (chroma, colour depth, profile) triple for each deep typed format.
fn shape_of(pf: PixelFormat) -> (ChromaFormat, BitDepth, Profile) {
    match pf {
        PixelFormat::Yuva422P10Le => (ChromaFormat::Y422, BitDepth::Ten, Profile::Hq),
        PixelFormat::Yuva422P12Le => (ChromaFormat::Y422, BitDepth::Twelve, Profile::Hq),
        PixelFormat::Yuva422P16Le => (ChromaFormat::Y422, BitDepth::Sixteen, Profile::Hq),
        PixelFormat::Yuva444P10Le => (ChromaFormat::Y444, BitDepth::Ten, Profile::Prores4444),
        PixelFormat::Yuva444P12Le => (ChromaFormat::Y444, BitDepth::Twelve, Profile::Prores4444),
        PixelFormat::Yuva444P16Le => (ChromaFormat::Y444, BitDepth::Sixteen, Profile::Prores4444),
        other => panic!("not a deep typed format: {other:?}"),
    }
}

const DEEP_FORMATS: [PixelFormat; 6] = [
    PixelFormat::Yuva422P10Le,
    PixelFormat::Yuva422P12Le,
    PixelFormat::Yuva422P16Le,
    PixelFormat::Yuva444P10Le,
    PixelFormat::Yuva444P12Le,
    PixelFormat::Yuva444P16Le,
];

/// The colour-only format at the same chroma/depth as a deep typed one.
fn untyped_twin(pf: PixelFormat) -> PixelFormat {
    match pf {
        PixelFormat::Yuva422P10Le => PixelFormat::Yuv422P10Le,
        PixelFormat::Yuva422P12Le => PixelFormat::Yuv422P12Le,
        PixelFormat::Yuva422P16Le => PixelFormat::Yuv422P16Le,
        PixelFormat::Yuva444P10Le => PixelFormat::Yuv444P10Le,
        PixelFormat::Yuva444P12Le => PixelFormat::Yuv444P12Le,
        PixelFormat::Yuva444P16Le => PixelFormat::Yuv444P16Le,
        other => panic!("not a deep typed format: {other:?}"),
    }
}

/// Encode a stream with 16-bit coded alpha via the established free
/// function; returns `(packet, source 16-bit alpha samples)`.
fn packet_with_16bit_alpha(chroma: ChromaFormat, profile: Profile) -> (Vec<u8>, Vec<u16>) {
    let (w, h) = (W as usize, H as usize);
    // 8-bit colour keeps the fixture small; the coded alpha width is
    // what matters here.
    let mut planes = match chroma {
        ChromaFormat::Y444 => vec![
            VideoPlane {
                stride: w,
                data: (0..w * h).map(|i| (32 + (i % 180)) as u8).collect(),
            },
            VideoPlane {
                stride: w,
                data: vec![128u8; w * h],
            },
            VideoPlane {
                stride: w,
                data: vec![128u8; w * h],
            },
        ],
        ChromaFormat::Y422 => vec![
            VideoPlane {
                stride: w,
                data: (0..w * h).map(|i| (24 + (i % 200)) as u8).collect(),
            },
            VideoPlane {
                stride: w / 2,
                data: vec![120u8; w / 2 * h],
            },
            VideoPlane {
                stride: w / 2,
                data: vec![136u8; w / 2 * h],
            },
        ],
    };
    let alpha = deep_alpha_samples(w, h, BitDepth::Sixteen);
    planes.push(le_plane(&alpha, w));
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        chroma,
        BitDepth::Eight,
        profile,
        profile.default_quant_index(),
        Some(AlphaChannelType::Sixteen),
    )
    .expect("encode with 16-bit alpha");
    (pkt, alpha)
}

// ─────────────────────────── decode surface ───────────────────────────

/// Every deep typed request agrees plane-for-plane with the untyped
/// same-depth request on an alpha-carrying stream — the typed surface
/// is routing plus a guarantee, not a separate decode path.
#[test]
fn deep_typed_decode_matches_untyped_path_on_alpha_stream() {
    for pf in DEEP_FORMATS {
        let (chroma, _depth, profile) = shape_of(pf);
        let (pkt, _) = packet_with_16bit_alpha(chroma, profile);
        let typed = decode_packet_with_format(&pkt, Some(0), Some(pf), OutputRange::Full)
            .expect("typed deep decode");
        let untyped =
            decode_packet_with_format(&pkt, Some(0), Some(untyped_twin(pf)), OutputRange::Full)
                .expect("untyped same-depth decode");
        assert_eq!(typed.planes.len(), 4, "{pf:?} must yield 4 planes");
        assert_eq!(
            untyped.planes.len(),
            4,
            "alpha stream: untyped twin appends the as-coded 4th plane"
        );
        for (i, (a, b)) in typed.planes.iter().zip(untyped.planes.iter()).enumerate() {
            assert_eq!(a.stride, b.stride, "{pf:?} plane {i} stride");
            assert_eq!(a.data, b.data, "{pf:?} plane {i} bytes");
        }
    }
}

/// The `*P16Le` surfaces deliver 16-bit coded alpha exactly: §7.1.2 is
/// lossless and the §7.5.2 conversion at `b = 16` is the identity, so
/// the decoded plane 3 is word-for-word the encoder's input.
#[test]
fn p16_surfaces_pass_16bit_coded_alpha_through_exactly() {
    for pf in [PixelFormat::Yuva422P16Le, PixelFormat::Yuva444P16Le] {
        let (chroma, _, profile) = shape_of(pf);
        let (pkt, src_alpha) = packet_with_16bit_alpha(chroma, profile);
        let (fh, _) = parse_frame(&pkt).expect("parse");
        assert_eq!(fh.alpha_channel_type, 2, "premise: 16-bit coded alpha");
        let typed = decode_packet_with_format(&pkt, Some(0), Some(pf), OutputRange::Full)
            .expect("typed P16 decode");
        assert_eq!(
            read_u16s(&typed.planes[3]),
            src_alpha,
            "{pf:?}: 16-bit alpha must survive encode → decode without any conversion"
        );
    }
}

/// 16-bit coded alpha lands on the 10-/12-bit surfaces demoted per
/// §7.5.2 — formula-exact, derivable from the lossless coded values.
#[test]
fn deep_surfaces_demote_16bit_coded_alpha_per_7_5_2() {
    for pf in [
        PixelFormat::Yuva444P10Le,
        PixelFormat::Yuva444P12Le,
        PixelFormat::Yuva422P10Le,
        PixelFormat::Yuva422P12Le,
    ] {
        let (chroma, depth, profile) = shape_of(pf);
        let (pkt, src_alpha) = packet_with_16bit_alpha(chroma, profile);
        let typed = decode_packet_with_format(&pkt, Some(0), Some(pf), OutputRange::Full)
            .expect("typed deep decode");
        let got = read_u16s(&typed.planes[3]);
        assert_eq!(got.len(), src_alpha.len());
        for (i, (&g, &s)) in got.iter().zip(src_alpha.iter()).enumerate() {
            let want = convert_752(s as u32, 65535, depth.max_value());
            assert_eq!(g, want, "{pf:?} alpha sample {i}: §7.5.2 demote mismatch");
        }
    }
}

/// 8-bit coded alpha promotes onto every deep surface per §7.5.2.
#[test]
fn deep_surfaces_promote_8bit_coded_alpha_per_7_5_2() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = deep_colour_planes(w, h, BitDepth::Twelve, ChromaFormat::Y444);
    let alpha8: Vec<u16> = deep_alpha_samples(w, h, BitDepth::Eight);
    planes.push(VideoPlane {
        stride: w,
        data: alpha8.iter().map(|&a| a as u8).collect(),
    });
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Twelve,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        Some(AlphaChannelType::Eight),
    )
    .expect("encode 12-bit colour + 8-bit alpha");
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert_eq!(fh.alpha_channel_type, 1, "premise: 8-bit coded alpha");

    for pf in [
        PixelFormat::Yuva444P10Le,
        PixelFormat::Yuva444P12Le,
        PixelFormat::Yuva444P16Le,
    ] {
        let (_, depth, _) = shape_of(pf);
        let typed = decode_packet_with_format(&pkt, Some(0), Some(pf), OutputRange::Full)
            .expect("typed deep decode");
        let got = read_u16s(&typed.planes[3]);
        for (i, (&g, &s)) in got.iter().zip(alpha8.iter()).enumerate() {
            let want = convert_752(s as u32, 255, depth.max_value());
            assert_eq!(g, want, "{pf:?} alpha sample {i}: §7.5.2 promote mismatch");
        }
    }
}

/// A stream that codes no alpha still honours the deep 4-plane
/// contract: plane 3 is synthesised fully opaque at the surface depth
/// (`2^b − 1` in every 16-bit LE word), full resolution.
#[test]
fn deep_typed_decode_synthesises_opaque_plane_at_depth() {
    let (w, h) = (W as usize, H as usize);
    for pf in DEEP_FORMATS {
        let (chroma, depth, profile) = shape_of(pf);
        let vf = VideoFrame {
            pts: Some(0),
            planes: deep_colour_planes(w, h, depth, chroma),
        };
        let pkt = encode_frame_with_alpha(
            &vf,
            W,
            H,
            chroma,
            depth,
            profile,
            profile.default_quant_index(),
            None,
        )
        .expect("encode without alpha");
        let (fh, _) = parse_frame(&pkt).expect("parse");
        assert_eq!(fh.alpha_channel_type, 0, "premise: no coded alpha");

        let typed = decode_packet_with_format(&pkt, Some(0), Some(pf), OutputRange::Full)
            .expect("typed deep decode");
        assert_eq!(typed.planes.len(), 4, "{pf:?} guarantees 4 planes");
        let a = &typed.planes[3];
        assert_eq!(a.stride, w * 2, "{pf:?}: full-res 16-bit-word alpha");
        assert_eq!(a.data.len(), w * h * 2);
        let max = depth.max_value() as u16;
        assert!(
            read_u16s(a).iter().all(|&s| s == max),
            "{pf:?}: synthesised alpha must be uniformly 2^b − 1 = {max}"
        );
    }
}

/// Chroma of a deep typed request must match the stream, exactly like
/// the colour-only requests.
#[test]
fn deep_typed_chroma_mismatch_rejected() {
    let (pkt_444, _) = packet_with_16bit_alpha(ChromaFormat::Y444, Profile::Prores4444);
    for pf in [
        PixelFormat::Yuva422P10Le,
        PixelFormat::Yuva422P12Le,
        PixelFormat::Yuva422P16Le,
    ] {
        let err = decode_packet_with_format(&pkt_444, Some(0), Some(pf), OutputRange::Full)
            .expect_err("4:2:2 deep request on a 4:4:4 stream must fail");
        assert!(
            err.to_string().contains("chroma"),
            "error should name the chroma mismatch: {err}"
        );
    }
}

/// The registry decoder honours a deep typed `CodecParameters` request.
#[test]
fn registry_decoder_honours_deep_typed_params() {
    use oxideav_core::{Packet, TimeBase};
    let (pkt, src_alpha) = packet_with_16bit_alpha(ChromaFormat::Y444, Profile::Prores4444);
    let mut dec =
        make_decoder(&params(PixelFormat::Yuva444P16Le)).expect("make_decoder Yuva444P16Le");
    let mut packet = Packet::new(0, TimeBase::new(1, 25), pkt);
    packet.pts = Some(0);
    dec.send_packet(&packet).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected a video frame");
    };
    assert_eq!(vf.planes.len(), 4);
    assert_eq!(read_u16s(&vf.planes[3]), src_alpha, "exact 16-bit alpha");
}

// ─────────────────────────── encode surface ───────────────────────────

/// A deep typed encoder's wire bytes are identical to the established
/// free-function path fed the equivalent input: the same colour planes
/// plus the alpha plane §7.5.2-promoted to 16-bit. This pins the typed
/// path as routing + conversion, with zero coder divergence.
#[test]
fn deep_typed_encode_is_byte_identical_to_promoted_free_function() {
    let (w, h) = (W as usize, H as usize);
    for pf in DEEP_FORMATS {
        let (chroma, depth, profile) = shape_of(pf);
        let colour = deep_colour_planes(w, h, depth, chroma);
        let alpha = deep_alpha_samples(w, h, depth);

        // Typed path: alpha plane at the format's depth. Pin the
        // profile explicitly so both routes quantise identically (the
        // bit-rate heuristic would pick Standard for 4:2:2 with no
        // bit_rate hint).
        let mut typed_planes = colour.clone();
        typed_planes.push(le_plane(&alpha, w));
        let cfg = EncoderConfig::default().with_profile(profile);
        let mut enc = make_encoder_with_config(&params(pf), cfg).expect("make_encoder deep typed");
        enc.send_frame(&frame_with(typed_planes))
            .expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");

        let (fh, _) = parse_frame(&pkt.data).expect("parse");
        assert_eq!(
            fh.alpha_channel_type, 2,
            "{pf:?}: deep typed input codes 16-bit alpha"
        );
        assert_eq!(fh.bitstream_version, 1, "alpha requires version 1 (§6.4)");

        // Free-function path: identical colour planes + the §7.5.2
        // promotion of the alpha samples to 16 bits.
        let promoted: Vec<u16> = alpha
            .iter()
            .map(|&a| convert_752(a as u32, depth.max_value(), 65535))
            .collect();
        let mut free_planes = colour;
        free_planes.push(le_plane(&promoted, w));
        let vf = VideoFrame {
            pts: Some(0),
            planes: free_planes,
        };
        let free = encode_frame_with_alpha(
            &vf,
            W,
            H,
            chroma,
            depth,
            profile,
            profile.default_quant_index(),
            Some(AlphaChannelType::Sixteen),
        )
        .expect("free-function encode with promoted alpha");
        assert_eq!(
            pkt.data, free,
            "{pf:?}: typed encoder must emit the exact bytes of the free function"
        );
    }
}

/// Deep typed alpha round-trips losslessly: encode at the format's
/// depth, decode at the same depth, get every sample back exactly (the
/// §7.5.2-mirror promotion is invertible by the §7.5.2 demotion).
#[test]
fn deep_typed_alpha_roundtrips_exactly() {
    let (w, h) = (W as usize, H as usize);
    for pf in DEEP_FORMATS {
        let (chroma, depth, _) = shape_of(pf);
        let mut planes = deep_colour_planes(w, h, depth, chroma);
        let alpha = deep_alpha_samples(w, h, depth);
        planes.push(le_plane(&alpha, w));

        let mut enc = make_encoder(&params(pf)).expect("make_encoder");
        enc.send_frame(&frame_with(planes)).expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");

        let out = decode_packet_with_format(&pkt.data, Some(0), Some(pf), OutputRange::Full)
            .expect("decode back");
        assert_eq!(
            read_u16s(&out.planes[3]),
            alpha,
            "{pf:?}: typed alpha must round-trip sample-exact"
        );
    }
}

/// 3-plane input under a deep typed format is refused with the
/// 4-plane-contract message.
#[test]
fn deep_typed_rejects_three_plane_frame() {
    let (w, h) = (W as usize, H as usize);
    let mut enc = make_encoder(&params(PixelFormat::Yuva444P12Le)).expect("make_encoder");
    let err = enc
        .send_frame(&frame_with(deep_colour_planes(
            w,
            h,
            BitDepth::Twelve,
            ChromaFormat::Y444,
        )))
        .expect_err("3-plane input under Yuva444P12Le must fail");
    assert!(
        err.to_string().contains("4-plane"),
        "error should explain the declared 4-plane contract: {err}"
    );
}

/// A 1-byte-per-sample alpha plane under a deep typed format is
/// refused — the format declares 16-bit LE words.
#[test]
fn deep_typed_rejects_one_byte_alpha_plane() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = deep_colour_planes(w, h, BitDepth::Ten, ChromaFormat::Y444);
    planes.push(VideoPlane {
        stride: w,
        data: vec![255u8; w * h],
    });
    let mut enc = make_encoder(&params(PixelFormat::Yuva444P10Le)).expect("make_encoder");
    let err = enc
        .send_frame(&frame_with(planes))
        .expect_err("1-byte alpha plane under Yuva444P10Le must fail");
    assert!(
        err.to_string().contains("16-bit-word"),
        "error should name the declared alpha layout: {err}"
    );
}

/// Explicit `AlphaChannelType::Eight` contradicts a deep typed format
/// (it would silently discard input precision) and is refused at
/// construction; explicit `Sixteen` is redundant but consistent.
#[test]
fn deep_typed_rejects_contradictory_alpha_config() {
    let cfg = EncoderConfig::default().with_alpha_channel_type(AlphaChannelType::Eight);
    let err = make_encoder_with_config(&params(PixelFormat::Yuva444P12Le), cfg)
        .err()
        .expect("deep typed + explicit 8-bit alpha coding must fail at construction");
    assert!(
        err.to_string().contains("16-bit alpha"),
        "error should explain the deep format's coded width: {err}"
    );
    let cfg = EncoderConfig::default().with_alpha_channel_type(AlphaChannelType::Sixteen);
    make_encoder_with_config(&params(PixelFormat::Yuva422P16Le), cfg)
        .expect("deep typed + explicit 16-bit alpha coding is consistent");
}

/// `output_params()` reports the deep typed format so muxers see the
/// declared alpha-carrying surface.
#[test]
fn deep_typed_encoder_output_params_report_typed_format() {
    for pf in DEEP_FORMATS {
        let enc = make_encoder(&params(pf)).expect("make_encoder");
        assert_eq!(
            enc.output_params().pixel_format,
            Some(pf),
            "output_params must carry the declared format"
        );
    }
}

/// Interlaced (TFF) deep typed round-trip through the registry config
/// path: the §6.2 field split, the §7.5.3 deinterleave, the typed
/// 4-plane guarantee, and the deep alpha conversion compose.
#[test]
fn deep_typed_interlaced_alpha_roundtrip() {
    let (w, h) = (W as usize, H as usize);
    let pf = PixelFormat::Yuva444P12Le;
    let (chroma, depth, _) = shape_of(pf);
    let mut planes = deep_colour_planes(w, h, depth, chroma);
    let alpha = deep_alpha_samples(w, h, depth);
    planes.push(le_plane(&alpha, w));

    let cfg = EncoderConfig::default().with_interlace_mode(1);
    let mut enc = make_encoder_with_config(&params(pf), cfg).expect("make_encoder interlaced");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.interlace_mode, 1, "premise: TFF stream");
    assert_eq!(fh.alpha_channel_type, 2, "premise: 16-bit coded alpha");

    let out = decode_packet_with_format(&pkt.data, Some(0), Some(pf), OutputRange::Full)
        .expect("decode interlaced deep typed");
    assert_eq!(out.planes.len(), 4);
    assert_eq!(
        read_u16s(&out.planes[3]),
        alpha,
        "interlaced deep typed alpha must round-trip sample-exact"
    );
}

/// Rate control carries the lossless deep alpha blob through every
/// trial encode.
#[test]
fn deep_typed_rate_control_carries_alpha() {
    let (w, h) = (W as usize, H as usize);
    let pf = PixelFormat::Yuva444P10Le;
    let (chroma, depth, _) = shape_of(pf);
    let mut planes = deep_colour_planes(w, h, depth, chroma);
    let alpha = deep_alpha_samples(w, h, depth);
    planes.push(le_plane(&alpha, w));

    let mut p = params(pf);
    p.bit_rate = Some(2_000_000);
    p.frame_rate = Some(Rational::new(25, 1));
    let cfg = EncoderConfig::default().with_rate_control();
    let mut enc = make_encoder_with_config(&p, cfg).expect("make_encoder rate-controlled");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 2);
    let out =
        decode_packet_with_format(&pkt.data, Some(0), Some(pf), OutputRange::Full).expect("decode");
    assert_eq!(
        read_u16s(&out.planes[3]),
        alpha,
        "every rate-control trial must carry the lossless alpha blob"
    );
}

// ─────────────────────── registry round trip ───────────────────────

/// Registry end-to-end at the 16-bit surface: encoder and decoder both
/// resolved from `CodecParameters { pixel_format: Yuva444P16Le }`;
/// alpha survives word-exact, colour round-trips at the §7.5.1 b = 16
/// surface.
#[test]
fn registry_deep_typed_end_to_end() {
    let (w, h) = (W as usize, H as usize);
    let pf = PixelFormat::Yuva444P16Le;
    let mut planes = deep_colour_planes(w, h, BitDepth::Sixteen, ChromaFormat::Y444);
    let src_y = read_u16s(&planes[0]);
    let alpha = deep_alpha_samples(w, h, BitDepth::Sixteen);
    planes.push(le_plane(&alpha, w));

    let mut reg = oxideav_core::CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);

    let mut enc = reg.first_encoder(&params(pf)).expect("registry encoder");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let mut dec = reg.first_decoder(&params(pf)).expect("registry decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected a video frame");
    };
    assert_eq!(vf.planes.len(), 4, "typed surface: 4 planes guaranteed");
    assert_eq!(read_u16s(&vf.planes[3]), alpha, "exact 16-bit alpha");

    // Colour is lossy; assert a sane 16-bit-domain PSNR on luma.
    let got_y = read_u16s(&vf.planes[0]);
    assert_eq!(got_y.len(), src_y.len());
    let mut mse = 0.0f64;
    for (&a, &b) in src_y.iter().zip(got_y.iter()) {
        let d = a as f64 - b as f64;
        mse += d * d;
    }
    mse /= src_y.len() as f64;
    let psnr = 10.0 * (65535.0f64 * 65535.0 / mse.max(1e-9)).log10();
    eprintln!("Yuva444P16Le luma PSNR = {psnr:.2} dB");
    assert!(psnr > 60.0, "16-bit luma PSNR too low: {psnr:.2} dB");
}

// ─────────── 16-bit colour surfaces (the untyped twins) ───────────

/// The colour-only 16-bit request round-trips and, on an alpha stream,
/// appends the as-coded 4th plane with exact 16-bit alpha — the
/// untyped equivalent of the `*P16Le` typed guarantee.
#[test]
fn untyped_16le_surface_roundtrips_and_carries_exact_alpha() {
    let (pkt, src_alpha) = packet_with_16bit_alpha(ChromaFormat::Y444, Profile::Prores4444);
    let out = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuv444P16Le),
        OutputRange::Full,
    )
    .expect("decode Yuv444P16Le");
    assert_eq!(
        out.planes.len(),
        4,
        "alpha stream: as-coded convention appends plane 3"
    );
    assert_eq!(
        read_u16s(&out.planes[3]),
        src_alpha,
        "16-bit coded alpha reaches the 16-bit as-coded surface exactly"
    );
    // Colour planes are 16-bit words spanning the full-scale range.
    assert_eq!(out.planes[0].stride, W as usize * 2);
    let y = read_u16s(&out.planes[0]);
    assert!(
        y.iter().any(|&v| v > 4095),
        "16-bit colour output should exceed the 12-bit ceiling somewhere"
    );
}

// ─────────── real reference bitstream through the deep surfaces ───────────

/// Decode the first frame of the in-tree `4444-with-alpha` reference
/// fixture (1920×1080 ap4h, 16-bit coded alpha, real third-party
/// encoder bytes) through `Yuva444P16Le` and pin byte-equality with
/// the untyped 16-bit decode — plus the §7.5.2 12↔16 relation against
/// the `Yuva444P12Le` surface on the same bytes.
///
/// Skips when the workspace `docs/` corpus is not checked out next to
/// the crate (standalone CI).
#[test]
fn deep_typed_surfaces_on_reference_fixture() {
    let path =
        std::path::PathBuf::from("../../docs/video/prores/fixtures/4444-with-alpha/input.mov");
    let mov = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip: missing {} ({e}). docs/ fixtures live in the workspace checkout only.",
                path.display()
            );
            return;
        }
    };
    // First ProRes frame container: scan for 'icpf', read the preceding
    // big-endian frame_size (RDD 36 §5.1).
    let mut frame_bytes: Option<&[u8]> = None;
    let mut i = 4usize;
    while i + 4 <= mov.len() {
        if &mov[i..i + 4] == b"icpf" {
            let size_off = i - 4;
            let fs = u32::from_be_bytes(mov[size_off..size_off + 4].try_into().unwrap()) as usize;
            if fs >= 8 && size_off + fs <= mov.len() {
                frame_bytes = Some(&mov[size_off..size_off + fs]);
                break;
            }
        }
        i += 1;
    }
    let pkt = frame_bytes.expect("no icpf frame in fixture container");

    let (fh, _) = parse_frame(pkt).expect("parse fixture frame");
    assert_eq!(fh.alpha_channel_type, 2, "fixture premise: 16-bit alpha");

    let typed16 = decode_packet_with_format(
        pkt,
        Some(0),
        Some(PixelFormat::Yuva444P16Le),
        OutputRange::Full,
    )
    .expect("typed P16 decode of reference fixture");
    let untyped16 = decode_packet_with_format(
        pkt,
        Some(0),
        Some(PixelFormat::Yuv444P16Le),
        OutputRange::Full,
    )
    .expect("untyped 16-bit decode");
    assert_eq!(typed16.planes.len(), 4);
    assert_eq!(untyped16.planes.len(), 4);
    for (i, (a, b)) in typed16
        .planes
        .iter()
        .zip(untyped16.planes.iter())
        .enumerate()
    {
        assert_eq!(a.stride, b.stride, "plane {i} stride");
        assert_eq!(a.data, b.data, "plane {i} bytes");
    }
    assert_eq!(
        typed16.planes[3].data.len(),
        1920 * 1080 * 2,
        "16-bit alpha plane size"
    );

    // The 12-bit surface's alpha is exactly the §7.5.2 demotion of the
    // 16-bit surface's (both derive from the same lossless coded
    // values).
    let typed12 = decode_packet_with_format(
        pkt,
        Some(0),
        Some(PixelFormat::Yuva444P12Le),
        OutputRange::Full,
    )
    .expect("typed P12 decode of reference fixture");
    let a16 = read_u16s(&typed16.planes[3]);
    let a12 = read_u16s(&typed12.planes[3]);
    for (i, (&hi, &lo)) in a16.iter().zip(a12.iter()).enumerate() {
        assert_eq!(
            lo,
            convert_752(hi as u32, 65535, 4095),
            "alpha sample {i}: 12-bit surface must be the §7.5.2 demotion of the 16-bit one"
        );
    }
}
