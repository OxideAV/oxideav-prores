//! `stuffing()` emission — RDD 36 §5.1.2 (syntax) + §6.1.2 (semantics).
//!
//! RDD 36 §5.1 places exactly one optional element after the last
//! `picture()` in a `frame()`: `stuffing()`, a run of `zero_byte`
//! (`0x00`) values used to "pad the compressed frame up to a desired
//! size" (§6.1.2 `zero_byte`). §6.1.2 fixes the relationship:
//!
//! ```text
//! frameDataSize = 4 + 4 + frame_header_size + picture("first").picture_size
//!               (+ picture("second").picture_size if interlaced)
//! stuffing_size = frame_size - frameDataSize          // >= 0
//! ```
//!
//! and §6.1.1 `frame_size` is "the total size of the compressed frame in
//! bytes (including the frame_size element itself and, if present,
//! stuffing)". So padding a frame appends `stuffing_size` zero bytes after
//! the picture(s) and rewrites the leading `frame_size` u32 to the padded
//! total.
//!
//! This crate exposes the pad two ways: the free function
//! [`encoder::pad_frame_to_size`] (post-process any assembled frame) and
//! the [`encoder::EncoderConfig::with_min_frame_size`] knob (threaded
//! through `make_encoder_with_config` → `send_frame`). The decoder already
//! consumes only the coded picture(s) and discards the trailing bytes
//! (RDD 36 §5.1), so a padded frame must decode bit-identically to its
//! unpadded twin.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::decoder::decode_packet;
use oxideav_prores::encoder::{
    encode_frame, make_encoder_with_config, pad_frame_to_size, EncoderConfig,
};
use oxideav_prores::frame::{parse_frame, ChromaFormat, Profile, FRAME_IDENTIFIER};
use oxideav_prores::CODEC_ID_STR;

/// Synthetic 4:2:2 gradient frame, large enough that the coded frame
/// is well under any reasonable pad target.
fn synth_422(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 7 + j * 5) % 200 + 32) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 3).clamp(-80, 80)) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 3).clamp(-80, 80)) as u8;
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

fn enc_params(width: u32, height: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.media_type = MediaType::Video;
    p.width = Some(width);
    p.height = Some(height);
    p.pixel_format = Some(PixelFormat::Yuv422P);
    p
}

/// Read the leading big-endian `frame_size` u32 (RDD 36 §5.1 `frame()`).
fn frame_size_field(pkt: &[u8]) -> u32 {
    u32::from_be_bytes(pkt[0..4].try_into().unwrap())
}

/// A coded frame already shorter than the target gains exactly
/// `target - coded_len` zero bytes, and the leading `frame_size` field is
/// rewritten to the padded total (RDD 36 §6.1.2: `frame_size` includes
/// "the frame_size element itself and, if present, stuffing").
#[test]
fn pad_grows_short_frame_and_rewrites_frame_size() {
    let frame = synth_422(128, 64);
    let coded = encode_frame(&frame, 128, 64, ChromaFormat::Y422, Profile::Standard, 4).unwrap();
    let coded_len = coded.len();
    // The §5.1 frame_size field on the unpadded frame equals the coded
    // length (no stuffing yet).
    assert_eq!(frame_size_field(&coded) as usize, coded_len);

    let target = (coded_len + 1024) as u32;
    let padded = pad_frame_to_size(&coded, target).unwrap();

    // Length is exactly the target.
    assert_eq!(padded.len(), target as usize);
    // §6.1.2: leading frame_size now carries the padded total.
    assert_eq!(frame_size_field(&padded), target);
    // The 'icpf' identifier and the entire coded prefix are byte-identical
    // (stuffing only appends; it never rewrites picture bytes).
    assert_eq!(&padded[4..8], FRAME_IDENTIFIER);
    assert_eq!(&padded[8..coded_len], &coded[8..]);
    // §5.1.2: the appended tail is all zero_byte (0x00).
    assert!(padded[coded_len..].iter().all(|&b| b == 0));
    assert_eq!(padded.len() - coded_len, 1024);
}

/// §6.1.2 `stuffing_size = frame_size − frameDataSize` is non-negative by
/// construction: a target at or below the coded size leaves the frame
/// byte-identical (no padding, no frame_size rewrite).
#[test]
fn pad_target_at_or_below_coded_size_is_noop() {
    let frame = synth_422(128, 64);
    let coded = encode_frame(&frame, 128, 64, ChromaFormat::Y422, Profile::Standard, 4).unwrap();
    let coded_len = coded.len() as u32;

    // Exactly equal → unchanged.
    let same = pad_frame_to_size(&coded, coded_len).unwrap();
    assert_eq!(same, coded);
    // Below → unchanged (stuffing never shrinks a frame).
    let below = pad_frame_to_size(&coded, coded_len.saturating_sub(50)).unwrap();
    assert_eq!(below, coded);
    // A target of zero is the degenerate below-case.
    let zero = pad_frame_to_size(&coded, 0).unwrap();
    assert_eq!(zero, coded);
}

/// A frame too short to even carry the 8-byte `frame_size` + `'icpf'`
/// preamble cannot be padded — reject cleanly rather than index out of
/// bounds.
#[test]
fn pad_rejects_truncated_frame() {
    for n in 0..8usize {
        let err = pad_frame_to_size(&vec![0u8; n], 100).expect_err("must reject");
        assert!(
            err.to_string().contains("frame too short"),
            "error must explain the truncation (n={n}, got: {err})"
        );
    }
}

/// A padded frame decodes bit-identically to its unpadded twin: the
/// decoder consumes only the coded picture(s) per §5.1 and discards the
/// trailing zero bytes.
#[test]
fn padded_frame_decodes_identically_to_unpadded() {
    let frame = synth_422(128, 64);
    let coded = encode_frame(&frame, 128, 64, ChromaFormat::Y422, Profile::Standard, 4).unwrap();
    let padded = pad_frame_to_size(&coded, (coded.len() + 4096) as u32).unwrap();

    let a = decode_packet(&coded, Some(0)).expect("decode unpadded");
    let b = decode_packet(&padded, Some(0)).expect("decode padded");

    assert_eq!(a.planes.len(), b.planes.len());
    for (pa, pb) in a.planes.iter().zip(b.planes.iter()) {
        assert_eq!(pa.stride, pb.stride);
        assert_eq!(pa.data, pb.data, "padded frame must decode identically");
    }

    // The parser also walks the padded frame without error and yields the
    // same frame header (the trailing stuffing is outside frame_header()).
    let (fh_a, _) = parse_frame(&coded).unwrap();
    let (fh_b, _) = parse_frame(&padded).unwrap();
    assert_eq!(fh_a.width, fh_b.width);
    assert_eq!(fh_a.height, fh_b.height);
}

/// `EncoderConfig::with_min_frame_size` threads through
/// `make_encoder_with_config` → `send_frame`: a registry-built encoder
/// pads every emitted packet up to the configured minimum.
#[test]
fn config_min_frame_size_pads_emitted_packets() {
    let params = enc_params(128, 64);

    // Baseline: unpadded coded size from the default config.
    let mut base_enc = make_encoder_with_config(&params, EncoderConfig::default()).unwrap();
    base_enc
        .send_frame(&Frame::Video(synth_422(128, 64)))
        .unwrap();
    let base_len = base_enc.receive_packet().unwrap().data.len();

    let target = (base_len + 2048) as u32;
    let cfg = EncoderConfig::default().with_min_frame_size(target);
    assert_eq!(cfg.min_frame_size, Some(target));

    let mut enc = make_encoder_with_config(&params, cfg).unwrap();
    enc.send_frame(&Frame::Video(synth_422(128, 64))).unwrap();
    let pkt = enc.receive_packet().unwrap();

    assert_eq!(pkt.data.len(), target as usize);
    assert_eq!(frame_size_field(&pkt.data), target);
    // The tail past the original coded length is all stuffing (0x00).
    assert!(pkt.data[base_len..].iter().all(|&b| b == 0));

    // And the padded packet still decodes.
    let decoded = decode_packet(&pkt.data, Some(0)).expect("decode padded packet");
    assert_eq!(decoded.planes.len(), 3);
}

/// `with_min_frame_size` below the natural coded size is a no-op: a config
/// asking for a tiny minimum emits exactly the unpadded bytes (the
/// per-frame stuffing_size would be negative, so none is added).
#[test]
fn config_min_frame_size_below_coded_is_noop() {
    let params = enc_params(128, 64);
    let mut base_enc = make_encoder_with_config(&params, EncoderConfig::default()).unwrap();
    base_enc
        .send_frame(&Frame::Video(synth_422(128, 64)))
        .unwrap();
    let base = base_enc.receive_packet().unwrap().data;

    let cfg = EncoderConfig::default().with_min_frame_size(8); // far below
    let mut enc = make_encoder_with_config(&params, cfg).unwrap();
    enc.send_frame(&Frame::Video(synth_422(128, 64))).unwrap();
    let pkt = enc.receive_packet().unwrap();

    assert_eq!(pkt.data, base, "tiny min_frame_size must not change bytes");
}
