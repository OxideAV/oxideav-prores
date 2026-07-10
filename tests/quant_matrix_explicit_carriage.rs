//! `EncoderConfig::explicit_qmat_carriage` — the both-tables frame
//! header form the reference streams always use.
//!
//! Every reference stream in the in-tree corpus carries BOTH
//! quantisation tables explicitly (`load_luma_quantization_matrix =
//! load_chroma_quantization_matrix = 1`, a 148-byte frame header), even
//! when the chroma table is a byte-for-byte copy of the luma table.
//! The encoder's default is the *minimal* RDD 36 §6.1.1 carriage
//! (`QuantMatrices::wire_flags`), which drops a redundant table. This
//! suite pins the opt-in explicit form:
//!
//! 1. flags `(1, 1)` + 148-byte header for every profile and matrix
//!    choice, with both tables on the wire verbatim in natural raster
//!    order — including the §6.1.1 chroma-copies-luma profiles and the
//!    flat all-4s default (all entries 4, legal per the `2..=63` range);
//! 2. decoded planes byte-identical to the minimal-carriage twin (the
//!    carriage form is semantically a no-op);
//! 3. the default config's output is byte-identical with and without
//!    the new field left at `false` (no SHA drift);
//! 4. form parity with the corpus: with the signature preset, the
//!    emitted flags byte and both wire tables equal the reference
//!    fixture's bytes at the same offsets (skips when `docs/` is absent).

use std::fs;
use std::path::PathBuf;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, Packet, PixelFormat, TimeBase,
    VideoFrame,
};
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{parse_frame, Profile};
use oxideav_prores::quant::{QuantMatrices, DEFAULT_QMAT};
use oxideav_prores::CODEC_ID_STR;

const W: u32 = 64;
const H: u32 = 48;

// Wire offsets within a frame unit (see tests/quant_matrix_order.rs).
const FH_SIZE_OFF: usize = 8;
const FLAGS_OFF: usize = 27;
const LUMA_TABLE: std::ops::Range<usize> = 28..92;
const CHROMA_TABLE: std::ops::Range<usize> = 92..156;

fn synth(is_444: bool) -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = if is_444 { w } else { w / 2 };
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (((i * 11 + j * 5) as u8) ^ ((i + 2 * j) as u8)).wrapping_add(32);
        }
        for i in 0..cw {
            cb[j * cw + i] = 90 + (((i * 3) ^ j) as u8 & 0x3F);
            cr[j * cw + i] = (160u8).wrapping_sub(((i + j * 2) as u8) & 0x3F);
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

fn params_for(profile: Profile) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(match profile {
        Profile::Prores4444 | Profile::Prores4444Xq => PixelFormat::Yuv444P,
        _ => PixelFormat::Yuv422P,
    });
    params
}

fn encode_with(config: EncoderConfig, profile: Profile) -> Vec<u8> {
    let params = params_for(profile);
    let mut enc =
        make_encoder_with_config(&params, config.with_profile(profile)).expect("make encoder");
    let is_444 = matches!(profile, Profile::Prores4444 | Profile::Prores4444Xq);
    enc.send_frame(&Frame::Video(synth(is_444))).expect("send");
    enc.receive_packet().expect("receive").data
}

fn decode_planes(pkt: &[u8], profile: Profile) -> Vec<VideoPlane> {
    let params = params_for(profile);
    let mut reg = CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make decoder");
    let mut p = Packet::new(0, TimeBase::new(1, 30), pkt.to_vec());
    p.flags.keyframe = true;
    dec.send_packet(&p).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v.planes,
        _ => panic!("expected video frame"),
    }
}

const ALL_PROFILES: [Profile; 6] = [
    Profile::Proxy,
    Profile::Lt,
    Profile::Standard,
    Profile::Hq,
    Profile::Prores4444,
    Profile::Prores4444Xq,
];

#[test]
fn explicit_carriage_ships_both_tables_verbatim_for_every_profile() {
    for profile in ALL_PROFILES {
        let qm = QuantMatrices::signature_for_profile(profile);
        let pkt = encode_with(
            EncoderConfig::signature_for_profile(profile).with_explicit_qmat_carriage(),
            profile,
        );
        let fh_size = u16::from_be_bytes([pkt[FH_SIZE_OFF], pkt[FH_SIZE_OFF + 1]]);
        assert_eq!(fh_size, 148, "{profile:?}: both tables → 148-byte header");
        assert_eq!(pkt[FLAGS_OFF] & 0b11, 0b11, "{profile:?}: flags (1, 1)");
        assert_eq!(&pkt[LUMA_TABLE], &qm.luma[..], "{profile:?}: luma table");
        assert_eq!(
            &pkt[CHROMA_TABLE],
            &qm.chroma[..],
            "{profile:?}: chroma table (explicit copy for the §6.1.1 \
             chroma-copies-luma profiles)"
        );
        // The parsed header reconstructs the same pair and reports the
        // explicit-chroma provenance.
        let (fh, _) = parse_frame(&pkt).expect("parse");
        assert!(fh.load_luma_quantization_matrix);
        assert!(fh.load_chroma_quantization_matrix);
        assert_eq!(QuantMatrices::from_header(&fh), qm);
    }
}

#[test]
fn explicit_carriage_with_flat_default_is_legal_and_verbatim() {
    // The all-4s default is within the §6.1.1 entry range, so shipping
    // it explicitly is legal; the wire carries two all-4 tables.
    let pkt = encode_with(
        EncoderConfig::flat().with_explicit_qmat_carriage(),
        Profile::Standard,
    );
    let fh_size = u16::from_be_bytes([pkt[FH_SIZE_OFF], pkt[FH_SIZE_OFF + 1]]);
    assert_eq!(fh_size, 148);
    assert_eq!(pkt[FLAGS_OFF] & 0b11, 0b11);
    assert_eq!(&pkt[LUMA_TABLE], &DEFAULT_QMAT[..]);
    assert_eq!(&pkt[CHROMA_TABLE], &DEFAULT_QMAT[..]);
}

#[test]
fn explicit_and_minimal_carriage_decode_byte_identically() {
    for profile in ALL_PROFILES {
        let minimal = encode_with(EncoderConfig::signature_for_profile(profile), profile);
        let explicit = encode_with(
            EncoderConfig::signature_for_profile(profile).with_explicit_qmat_carriage(),
            profile,
        );
        if QuantMatrices::signature_for_profile(profile).wire_flags() == (true, true) {
            // Proxy's minimal carriage is already both-tables, so the
            // explicit form coincides with it byte-for-byte.
            assert_eq!(
                minimal, explicit,
                "{profile:?}: minimal carriage is already (1, 1)"
            );
        } else {
            assert_ne!(
                minimal.len(),
                explicit.len(),
                "{profile:?}: carriage forms must differ on the wire"
            );
        }
        let p_min = decode_planes(&minimal, profile);
        let p_exp = decode_planes(&explicit, profile);
        assert_eq!(p_min.len(), p_exp.len());
        for (i, (a, b)) in p_min.iter().zip(p_exp.iter()).enumerate() {
            assert_eq!(
                a.data, b.data,
                "{profile:?} plane {i}: carriage form must not change pixels"
            );
        }
    }

    // Flat explicit twin decodes identically to the legacy no-tables form.
    let minimal = encode_with(EncoderConfig::flat(), Profile::Standard);
    let explicit = encode_with(
        EncoderConfig::flat().with_explicit_qmat_carriage(),
        Profile::Standard,
    );
    let p_min = decode_planes(&minimal, Profile::Standard);
    let p_exp = decode_planes(&explicit, Profile::Standard);
    for (i, (a, b)) in p_min.iter().zip(p_exp.iter()).enumerate() {
        assert_eq!(a.data, b.data, "flat twin plane {i}");
    }
}

#[test]
fn default_config_output_is_unchanged() {
    // The new field defaults to false; a default-config encode must be
    // byte-identical to an explicit `explicit_qmat_carriage: false`
    // construction — no drift in existing output.
    let a = encode_with(EncoderConfig::default(), Profile::Standard);
    let b = encode_with(
        EncoderConfig {
            explicit_qmat_carriage: false,
            ..EncoderConfig::default()
        },
        Profile::Standard,
    );
    assert_eq!(a, b);
    let fh_size = u16::from_be_bytes([a[FH_SIZE_OFF], a[FH_SIZE_OFF + 1]]);
    assert_eq!(fh_size, 20, "default flat config keeps the 20-byte header");
    assert_eq!(a[FLAGS_OFF] & 0b11, 0, "default flat config keeps (0, 0)");
}

#[test]
fn explicit_carriage_matches_reference_fixture_header_form() {
    // With the signature preset + explicit carriage, the flags byte and
    // both wire tables must equal the reference fixture's bytes at the
    // same offsets. Skips when the docs/ corpus is absent (standalone CI).
    let path = PathBuf::from("../../docs/video/prores/fixtures/sq-1920x1080/input.mov");
    let container = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: missing {} ({e})", path.display());
            return;
        }
    };
    // Locate the raw frame inside the container.
    let mut fixture_frame: Option<Vec<u8>> = None;
    let mut i = 4usize;
    while i + 4 <= container.len() {
        if &container[i..i + 4] == b"icpf" {
            let size_off = i - 4;
            let frame_size =
                u32::from_be_bytes(container[size_off..size_off + 4].try_into().unwrap()) as usize;
            if size_off + frame_size <= container.len() && frame_size >= 8 {
                fixture_frame = Some(container[size_off..size_off + frame_size].to_vec());
                break;
            }
        }
        i += 1;
    }
    let fixture = fixture_frame.expect("no ProRes frame in fixture container");

    let ours = encode_with(
        EncoderConfig::signature_for_profile(Profile::Standard).with_explicit_qmat_carriage(),
        Profile::Standard,
    );
    assert_eq!(
        u16::from_be_bytes([ours[FH_SIZE_OFF], ours[FH_SIZE_OFF + 1]]),
        u16::from_be_bytes([fixture[FH_SIZE_OFF], fixture[FH_SIZE_OFF + 1]]),
        "frame_header_size parity with the reference stream"
    );
    assert_eq!(
        ours[FLAGS_OFF] & 0b11,
        fixture[FLAGS_OFF] & 0b11,
        "carriage flags parity with the reference stream"
    );
    assert_eq!(&ours[LUMA_TABLE], &fixture[LUMA_TABLE], "luma table bytes");
    assert_eq!(
        &ours[CHROMA_TABLE], &fixture[CHROMA_TABLE],
        "chroma table bytes"
    );
}
