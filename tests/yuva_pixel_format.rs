//! Alpha-typed pixel-format surface (`PixelFormat::Yuva422P` /
//! `Yuva444P`) on both codec directions.
//!
//! Before these formats existed in `oxideav-core`, decoded RDD 36 alpha
//! was reachable only as an *untyped* 4th plane: the caller requested
//! `Yuv4(2|4)4P*` and had to probe `frame.planes.len() == 4` to learn
//! whether alpha rode along, and no `CodecParameters::pixel_format`
//! value could *declare* an alpha-carrying stream on either the encode
//! or decode side. These tests pin the typed surface end-to-end:
//!
//! * **Decode** — `decode_packet_with_format(Some(Yuva444P | Yuva422P))`
//!   and the registry path (`make_decoder` with the same
//!   `CodecParameters`) guarantee a 4-plane 8-bit frame: coded alpha
//!   rides plane 3 (16-bit coded alpha demoted per RDD 36 §7.5.2,
//!   consistent with the §7.5.1 demotion of Y/Cb/Cr at 8-bit output),
//!   and a no-alpha stream gets a synthesised fully-opaque plane.
//! * **Encode** — `make_encoder` built with a `Yuva*` pixel format
//!   requires 4-plane input, codes alpha on every frame
//!   (`alpha_channel_type = 1`, bitstream version 1 per §6.4), and emits
//!   wire bytes identical to the pre-existing free-function /
//!   auto-detect paths.
//! * **Contract violations** — 3-plane input under a `Yuva*` format, a
//!   2-bytes-per-sample alpha plane under a `Yuva*` format, an explicit
//!   `AlphaChannelType::Sixteen` config on a `Yuva*` encoder, chroma
//!   mismatches, and the still-unsupported `PixelFormat` values all
//!   surface clean `Err`s.
//!
//! Alpha is coded losslessly per §7.1.2, so alpha assertions are
//! byte-exact, never a PSNR bar.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, VideoFrame};
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::decoder::{
    decode_packet, decode_packet_with_format, make_decoder, OutputRange,
};
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

/// Encode a 4444 packet with an 8-bit alpha plane via the free
/// function; returns `(packet, source_alpha_bytes)`.
fn packet_4444_alpha8() -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        Some(AlphaChannelType::Eight),
    )
    .expect("encode 4444 + 8-bit alpha");
    (pkt, src_alpha)
}

// ─────────────────────────── decode surface ───────────────────────────

#[test]
fn yuva444_decode_matches_untyped_path_on_alpha_stream() {
    let (pkt, src_alpha) = packet_4444_alpha8();

    let typed = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Full,
    )
    .expect("decode Yuva444P");
    let untyped = decode_packet(&pkt, Some(0)).expect("decode default 8-bit");

    assert_eq!(typed.planes.len(), 4, "Yuva444P must yield 4 planes");
    assert_eq!(
        typed.planes.len(),
        untyped.planes.len(),
        "alpha stream: typed and untyped requests agree on plane count"
    );
    for (i, (a, b)) in typed.planes.iter().zip(untyped.planes.iter()).enumerate() {
        assert_eq!(a.stride, b.stride, "plane {i} stride");
        assert_eq!(a.data, b.data, "plane {i} bytes");
    }
    assert_eq!(
        typed.planes[3].data, src_alpha,
        "8-bit alpha at the 8-bit surface is the lossless §7.1.2 identity"
    );
}

#[test]
fn yuva444_decode_synthesises_opaque_plane_on_no_alpha_stream() {
    let (w, h) = (W as usize, H as usize);
    let vf = VideoFrame {
        pts: Some(0),
        planes: yuv444_8(w, h),
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        None,
    )
    .expect("encode 4444 without alpha");
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert_eq!(fh.alpha_channel_type, 0, "premise: stream codes no alpha");

    let typed = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Full,
    )
    .expect("decode Yuva444P");
    let untyped =
        decode_packet_with_format(&pkt, Some(0), Some(PixelFormat::Yuv444P), OutputRange::Full)
            .expect("decode Yuv444P");

    assert_eq!(untyped.planes.len(), 3, "no coded alpha → untyped stays 3");
    assert_eq!(typed.planes.len(), 4, "Yuva444P guarantees 4 planes");
    for i in 0..3 {
        assert_eq!(
            typed.planes[i].data, untyped.planes[i].data,
            "plane {i}: colour planes must be unaffected by the alpha surface"
        );
    }
    let a = &typed.planes[3];
    assert_eq!(a.stride, w, "full-resolution 8-bit alpha stride");
    assert_eq!(a.data.len(), w * h);
    assert!(
        a.data.iter().all(|&s| s == 0xFF),
        "synthesised alpha must be fully opaque (§7.5.2 full-opacity code 255)"
    );
}

#[test]
fn yuva422_decode_carries_alpha_on_v1_422_stream() {
    // 4:2:2 + alpha is a legal bitstream-version-1 combination (§6.4).
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv422_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Hq,
        Profile::Hq.default_quant_index(),
        Some(AlphaChannelType::Eight),
    )
    .expect("encode 4:2:2 + alpha");
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert_eq!(fh.bitstream_version, 1, "4:2:2 + alpha must be version 1");

    let typed = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva422P),
        OutputRange::Full,
    )
    .expect("decode Yuva422P");
    assert_eq!(typed.planes.len(), 4);
    assert_eq!(typed.planes[1].stride, w / 2, "half-width chroma");
    assert_eq!(typed.planes[3].stride, w, "full-resolution alpha");
    assert_eq!(typed.planes[3].data, src_alpha, "lossless alpha round-trip");
}

#[test]
fn yuva444_demotes_16bit_coded_alpha_per_7_5_2() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_16(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444Xq,
        Profile::Prores4444Xq.default_quant_index(),
        Some(AlphaChannelType::Sixteen),
    )
    .expect("encode 4444 XQ + 16-bit alpha");
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert_eq!(fh.alpha_channel_type, 2, "premise: 16-bit coded alpha");

    let typed = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Full,
    )
    .expect("decode Yuva444P");
    assert_eq!(typed.planes.len(), 4);
    assert_eq!(
        typed.planes[3].data.len(),
        w * h,
        "16-bit coded alpha lands on the 8-bit surface as 1 byte/sample"
    );
    // The coded stream is lossless (§7.1.2), so the §7.5.2 demote
    // `alphaSample = round(255 * alpha / 65535)` is exactly derivable
    // from the source plane.
    for (i, &got) in typed.planes[3].data.iter().enumerate() {
        let src = u16::from_le_bytes([src_alpha[i * 2], src_alpha[i * 2 + 1]]);
        let want = ((src as u64 * 255 + 32767) / 65535) as u8;
        assert_eq!(got, want, "alpha sample {i}: §7.5.2 demote mismatch");
    }
}

#[test]
fn yuva_chroma_mismatch_rejected() {
    let (pkt_444, _) = packet_4444_alpha8();
    let err = decode_packet_with_format(
        &pkt_444,
        Some(0),
        Some(PixelFormat::Yuva422P),
        OutputRange::Full,
    )
    .expect_err("Yuva422P request on a 4:4:4 stream must fail");
    assert!(
        err.to_string().contains("chroma"),
        "error should name the chroma mismatch: {err}"
    );

    let (w, h) = (W as usize, H as usize);
    let vf = VideoFrame {
        pts: Some(0),
        planes: yuv422_8(w, h),
    };
    let pkt_422 = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        Profile::Standard.default_quant_index(),
        None,
    )
    .expect("encode 4:2:2");
    decode_packet_with_format(
        &pkt_422,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Full,
    )
    .expect_err("Yuva444P request on a 4:2:2 stream must fail");
}

#[test]
fn unsupported_pixel_formats_still_rejected() {
    let (pkt, _) = packet_4444_alpha8();
    for pf in [
        PixelFormat::Rgb24,
        PixelFormat::Yuv420P,
        PixelFormat::Yuva420P,
        PixelFormat::Yuv444P16Le,
    ] {
        decode_packet_with_format(&pkt, Some(0), Some(pf), OutputRange::Full)
            .expect_err("non-ProRes pixel format request must fail");
        assert!(
            make_decoder(&params(pf)).is_err(),
            "registry decoder must refuse {pf:?} too"
        );
        assert!(
            make_encoder(&params(pf)).is_err(),
            "registry encoder must refuse {pf:?} too"
        );
    }
}

#[test]
fn registry_decoder_honours_yuva444_params() {
    use oxideav_core::{Packet, TimeBase};
    let (pkt, src_alpha) = packet_4444_alpha8();

    let mut dec = make_decoder(&params(PixelFormat::Yuva444P)).expect("make_decoder Yuva444P");
    let mut packet = Packet::new(0, TimeBase::new(1, 25), pkt);
    packet.pts = Some(0);
    dec.send_packet(&packet).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected a video frame");
    };
    assert_eq!(vf.planes.len(), 4);
    assert_eq!(vf.planes[3].data, src_alpha);
}

// ─────────────────────────── encode surface ───────────────────────────

#[test]
fn yuva444_params_encode_is_byte_identical_to_free_function() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    planes.push(alpha_plane_8(w, h));

    let mut enc = make_encoder(&params(PixelFormat::Yuva444P)).expect("make_encoder Yuva444P");
    enc.send_frame(&frame_with(planes.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 1, "Yuva input codes 8-bit alpha");
    assert_eq!(fh.bitstream_version, 1, "alpha requires version 1 (§6.4)");
    assert_eq!(fh.chroma_format, ChromaFormat::Y444);

    // Byte-equivalence with the established free-function path — the
    // typed surface is routing, not a new coder.
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let free = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        Some(AlphaChannelType::Eight),
    )
    .expect("free-function encode");
    assert_eq!(
        pkt.data, free,
        "Yuva444P-typed encoder must emit the exact bytes of the free function"
    );
}

#[test]
fn yuva422_params_encode_codes_alpha_v1() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv422_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);

    let mut enc = make_encoder(&params(PixelFormat::Yuva422P)).expect("make_encoder Yuva422P");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.chroma_format, ChromaFormat::Y422);
    assert_eq!(fh.alpha_channel_type, 1);
    assert_eq!(fh.bitstream_version, 1, "4:2:2 + alpha must be version 1");

    let out = decode_packet_with_format(
        &pkt.data,
        Some(0),
        Some(PixelFormat::Yuva422P),
        OutputRange::Full,
    )
    .expect("decode back");
    assert_eq!(out.planes[3].data, src_alpha, "lossless alpha round-trip");
}

#[test]
fn yuva_params_reject_three_plane_frame() {
    let (w, h) = (W as usize, H as usize);
    let mut enc = make_encoder(&params(PixelFormat::Yuva444P)).expect("make_encoder");
    let err = enc
        .send_frame(&frame_with(yuv444_8(w, h)))
        .expect_err("3-plane input under Yuva444P must fail");
    assert!(
        err.to_string().contains("4-plane"),
        "error should explain the declared 4-plane contract: {err}"
    );
}

#[test]
fn yuva_params_reject_explicit_sixteen_config() {
    let cfg = EncoderConfig::default().with_alpha_channel_type(AlphaChannelType::Sixteen);
    let err = make_encoder_with_config(&params(PixelFormat::Yuva444P), cfg)
        .err()
        .expect("Yuva444P + explicit 16-bit alpha coding must fail at construction");
    assert!(
        err.to_string().contains("8-bit alpha"),
        "error should explain the 8-bit alpha contract: {err}"
    );
    // Explicit Eight is redundant but consistent — accepted.
    let cfg = EncoderConfig::default().with_alpha_channel_type(AlphaChannelType::Eight);
    make_encoder_with_config(&params(PixelFormat::Yuva422P), cfg)
        .expect("Yuva422P + explicit 8-bit alpha coding is consistent");
}

#[test]
fn yuva_params_reject_two_byte_alpha_plane() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    planes.push(alpha_plane_16(w, h));
    let mut enc = make_encoder(&params(PixelFormat::Yuva444P)).expect("make_encoder");
    let err = enc
        .send_frame(&frame_with(planes))
        .expect_err("2-bytes-per-sample alpha under Yuva444P must fail");
    assert!(
        err.to_string().contains("8-bit alpha"),
        "error should name the declared alpha depth: {err}"
    );
}

#[test]
fn yuva444_rate_control_carries_alpha() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);

    let mut p = params(PixelFormat::Yuva444P);
    p.bit_rate = Some(2_000_000);
    p.frame_rate = Some(Rational::new(25, 1));
    let cfg = EncoderConfig::default().with_rate_control();
    let mut enc = make_encoder_with_config(&p, cfg).expect("make_encoder rate-controlled");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let (fh, _) = parse_frame(&pkt.data).expect("parse");
    assert_eq!(fh.alpha_channel_type, 1);
    let out = decode_packet(&pkt.data, Some(0)).expect("decode");
    assert_eq!(
        out.planes[3].data, src_alpha,
        "every rate-control trial must carry the lossless alpha blob"
    );
}

// ─────────────────────── registry round trip ───────────────────────

#[test]
fn registry_yuva444_end_to_end() {
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);

    let mut reg = oxideav_core::CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);

    let mut enc = reg
        .first_encoder(&params(PixelFormat::Yuva444P))
        .expect("registry encoder for Yuva444P");
    enc.send_frame(&frame_with(planes)).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");

    let mut dec = reg
        .first_decoder(&params(PixelFormat::Yuva444P))
        .expect("registry decoder for Yuva444P");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected a video frame");
    };
    assert_eq!(vf.planes.len(), 4, "typed surface: 4 planes guaranteed");
    assert_eq!(vf.planes[0].data.len(), w * h);
    assert_eq!(vf.planes[3].data, src_alpha, "lossless alpha round-trip");
}

// ─────────────── interlaced streams through the typed surface ───────────────

/// Interlaced (TFF) 4:4:4 + 8-bit alpha through `Yuva444P`: the §6.2
/// field split, the §7.5.3 deinterleave, and the typed 4-plane
/// guarantee compose; alpha still round-trips byte-exactly.
#[test]
fn yuva444_interlaced_alpha_roundtrip() {
    use oxideav_prores::encoder::encode_frame_interlaced;
    let (w, h) = (W as usize, H as usize);
    let mut planes = yuv444_8(w, h);
    let alpha = alpha_plane_8(w, h);
    let src_alpha = alpha.data.clone();
    planes.push(alpha);
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_interlaced(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        Some(AlphaChannelType::Eight),
        1, // top-field-first
    )
    .expect("encode interlaced 4444 + alpha");
    let (fh, _) = parse_frame(&pkt).expect("parse");
    assert_eq!(fh.interlace_mode, 1, "premise: TFF stream");
    assert_eq!(fh.alpha_channel_type, 1, "premise: 8-bit alpha");

    let typed = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Full,
    )
    .expect("decode Yuva444P interlaced");
    assert_eq!(typed.planes.len(), 4);
    assert_eq!(typed.planes[3].data, src_alpha, "lossless alpha round-trip");
}

/// Interlaced (BFF) no-alpha stream through `Yuva444P`: the opaque
/// plane synthesis must hold on the interlaced decode path too.
#[test]
fn yuva444_interlaced_no_alpha_synthesises_opaque_plane() {
    use oxideav_prores::encoder::encode_frame_interlaced;
    let (w, h) = (W as usize, H as usize);
    let vf = VideoFrame {
        pts: Some(0),
        planes: yuv444_8(w, h),
    };
    let pkt = encode_frame_interlaced(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        None,
        2, // bottom-field-first
    )
    .expect("encode interlaced 4444 without alpha");
    let typed = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Full,
    )
    .expect("decode Yuva444P interlaced no-alpha");
    assert_eq!(typed.planes.len(), 4, "typed surface: 4 planes guaranteed");
    assert!(
        typed.planes[3].data.iter().all(|&s| s == 0xFF),
        "synthesised alpha must be fully opaque"
    );
    assert_eq!(typed.planes[3].data.len(), w * h);
}

// ─────────────── real reference bitstream through the typed surface ───────────────

/// Decode the first frame of the in-tree `4444-with-alpha` reference
/// fixture (1920×1080 ap4h, 16-bit coded alpha, real third-party
/// encoder bytes) through `Yuva444P` and pin byte-equality with the
/// untyped 8-bit decode — the typed surface is pure routing on top of
/// the same decode core, on real-world bytes as well as synthetic ones.
///
/// Skips when the workspace `docs/` corpus is not checked out next to
/// the crate (standalone CI).
#[test]
fn yuva444_typed_surface_on_reference_fixture() {
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
    assert_eq!(fh.chroma_format, ChromaFormat::Y444);

    let typed =
        decode_packet_with_format(pkt, Some(0), Some(PixelFormat::Yuva444P), OutputRange::Full)
            .expect("typed decode of reference fixture");
    let untyped = decode_packet(pkt, Some(0)).expect("untyped 8-bit decode");
    assert_eq!(typed.planes.len(), 4);
    assert_eq!(untyped.planes.len(), 4);
    for (i, (a, b)) in typed.planes.iter().zip(untyped.planes.iter()).enumerate() {
        assert_eq!(a.stride, b.stride, "plane {i} stride");
        assert_eq!(a.data, b.data, "plane {i} bytes");
    }
    assert_eq!(typed.planes[0].data.len(), 1920 * 1080, "8-bit luma size");
    assert_eq!(typed.planes[3].data.len(), 1920 * 1080, "8-bit alpha size");
}

// ─────────────── §7.5.1 Video clamp × typed surface interplay ───────────────

/// `OutputRange::Video` confines Y/Cb/Cr to `1..=254` at 8-bit output,
/// but §7.5.2 always maps decoded alpha across the full opacity range —
/// the typed surface must keep alpha byte-exact (extreme codes 0 and
/// 255 included) while the colour planes clamp.
#[test]
fn yuva444_video_range_clamps_colour_not_alpha() {
    let (w, h) = (W as usize, H as usize);
    // Push luma to both extremes so the Video clamp is observable, and
    // give alpha both extreme codes.
    let mut y = vec![0u8; w * h];
    for (i, v) in y.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 0 } else { 255 };
    }
    let planes = vec![
        VideoPlane { stride: w, data: y },
        VideoPlane {
            stride: w,
            data: vec![128u8; w * h],
        },
        VideoPlane {
            stride: w,
            data: vec![128u8; w * h],
        },
        {
            let mut a = alpha_plane_8(w, h);
            a.data[0] = 0;
            a.data[w * h - 1] = 255;
            a
        },
    ];
    let src_alpha = planes[3].data.clone();
    let vf = VideoFrame {
        pts: Some(0),
        planes,
    };
    let pkt = encode_frame_with_alpha(
        &vf,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Eight,
        Profile::Prores4444,
        Profile::Prores4444.default_quant_index(),
        Some(AlphaChannelType::Eight),
    )
    .expect("encode 4444 + alpha extremes");

    let video = decode_packet_with_format(
        &pkt,
        Some(0),
        Some(PixelFormat::Yuva444P),
        OutputRange::Video,
    )
    .expect("typed decode, Video range");
    assert_eq!(video.planes.len(), 4);
    for (p, name) in video.planes.iter().take(3).zip(["Y", "Cb", "Cr"]) {
        assert!(
            p.data.iter().all(|&s| (1..=254).contains(&s)),
            "{name} plane must clamp to the permissible video levels 1..=254"
        );
    }
    assert!(
        video.planes[0].data.contains(&1) && video.planes[0].data.contains(&254),
        "premise: the source extremes actually hit the Video clamp bounds"
    );
    assert_eq!(
        video.planes[3].data, src_alpha,
        "alpha is unaffected by the Video clamp (§7.5.2 full opacity range)"
    );
    assert_eq!(video.planes[3].data[0], 0, "extreme alpha code 0 survives");
    assert_eq!(
        video.planes[3].data[w * h - 1],
        255,
        "extreme alpha code 255 survives"
    );
}
