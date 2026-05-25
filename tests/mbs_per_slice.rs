//! `EncoderConfig::with_mbs_per_slice` knob (RDD 36 §5.3).
//!
//! Covers the encoder API addition that lets callers configure the
//! per-slice macroblock count. The spec stores
//! `log2_desired_slice_size_in_mb` in two bits of the picture header so
//! only `{1, 2, 4, 8}` MBs/slice are representable; the default is `8`
//! which matches every reference Apple-encoded fixture committed under
//! `docs/video/prores/fixtures/`.
//!
//! Lowering the value subdivides each macroblock row into more, smaller
//! slices; the per-slice fixed cost (`slice_header` + per-component
//! entropy coder reset + `slice_size_table` entry) is amortised over
//! fewer macroblocks so the encoded packet grows in a monotonic way as
//! the slice count rises. ffmpeg's `prores_ks` exposes the same knob
//! through `-mbs_per_slice {1,2,4,8}` and the spec documents the
//! reproduceability via `picture_header.log2_desired_slice_size_in_mb`.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::decoder::decode_packet;
use oxideav_prores::encoder::{
    make_encoder_with_config, mbs_per_slice_to_log2, EncoderConfig, DEFAULT_MBS_PER_SLICE,
};
use oxideav_prores::frame::{parse_frame, parse_picture_header};
use oxideav_prores::CODEC_ID_STR;

/// Synthetic 4:2:2 frame with a gradient that exercises every block
/// (so per-slice byte counts respond to the slice width).
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

/// Encode one frame with the supplied slice-width knob and return the
/// packet bytes (Vec<u8>).
fn encode_at_mbs_per_slice(width: u32, height: u32, mbs_per_slice: Option<u8>) -> Vec<u8> {
    let params = enc_params(width, height);
    let mut cfg = EncoderConfig::default();
    if let Some(m) = mbs_per_slice {
        cfg = cfg.with_mbs_per_slice(m);
    }
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
    enc.send_frame(&Frame::Video(synth_422(width, height)))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    pkt.data
}

/// Read the picture-header's `log2_desired_slice_size_in_mb` field
/// out of the first picture of an encoded ProRes packet.
fn read_log2_slice_mb_width(pkt: &[u8]) -> u8 {
    let (_, pictures) = parse_frame(pkt).expect("parse_frame");
    let (ph, _) = parse_picture_header(pictures).expect("parse_picture_header");
    ph.log2_desired_slice_size_in_mb
}

/// `DEFAULT_MBS_PER_SLICE` is 8 (matches every reference fixture under
/// `docs/video/prores/fixtures/`, all of which carry
/// `log2_slice_mb_width=3` per RDD 36 §5.3 / Table 2).
#[test]
fn default_mbs_per_slice_is_eight() {
    assert_eq!(DEFAULT_MBS_PER_SLICE, 8);
    assert_eq!(EncoderConfig::default().mbs_per_slice, None);
}

/// `mbs_per_slice_to_log2` accepts every legal value and rejects the
/// rest. The legal set `{1, 2, 4, 8}` matches the bit width of the
/// `log2_desired_slice_size_in_mb` picture-header field (2 bits) per
/// RDD 36 §5.3.
#[test]
fn mbs_per_slice_to_log2_accepts_powers_of_two_only() {
    assert_eq!(mbs_per_slice_to_log2(1).unwrap(), 0);
    assert_eq!(mbs_per_slice_to_log2(2).unwrap(), 1);
    assert_eq!(mbs_per_slice_to_log2(4).unwrap(), 2);
    assert_eq!(mbs_per_slice_to_log2(8).unwrap(), 3);
    for bad in [0u8, 3, 5, 6, 7, 9, 16, 32, 64, 100, 255] {
        let err = mbs_per_slice_to_log2(bad).expect_err("must reject");
        assert!(
            err.to_string().contains("mbs_per_slice"),
            "error must name mbs_per_slice (bad={bad}, got: {err})"
        );
    }
}

/// Every legal value `{1, 2, 4, 8}` should construct a valid encoder
/// and emit a packet whose `picture_header.log2_desired_slice_size_in_mb`
/// matches the request (RDD 36 §5.3 round-trip).
#[test]
fn mbs_per_slice_writes_picture_header_field() {
    for &m in &[1u8, 2, 4, 8] {
        let pkt = encode_at_mbs_per_slice(128, 64, Some(m));
        let log2 = read_log2_slice_mb_width(&pkt);
        let expected = mbs_per_slice_to_log2(m).unwrap();
        assert_eq!(
            log2, expected,
            "mbs_per_slice={m}: expected log2={expected} in picture header, got {log2}"
        );
    }
}

/// Non-power-of-two / out-of-range values must be rejected at
/// `make_encoder_with_config` time, not silently clamped.
#[test]
fn mbs_per_slice_invalid_rejected_at_construction() {
    let params = enc_params(64, 48);
    for &bad in &[0u8, 3, 5, 6, 7, 9, 16, 32] {
        let cfg = EncoderConfig::default().with_mbs_per_slice(bad);
        let err = make_encoder_with_config(&params, cfg)
            .err()
            .unwrap_or_else(|| panic!("mbs_per_slice={bad} must be rejected"));
        assert!(
            err.to_string().contains("mbs_per_slice"),
            "error must mention mbs_per_slice (bad={bad}, got: {err})"
        );
    }
}

/// Every legal slice width must self-roundtrip: encode → decode →
/// compare. The decoder reads `log2_desired_slice_size_in_mb` from the
/// picture header and rebuilds the per-row template via
/// `compute_slice_sizes`, so all four values must work end-to-end.
#[test]
fn mbs_per_slice_self_roundtrip_all_values() {
    let width = 128u32;
    let height = 64u32;
    let src = synth_422(width, height);
    for &m in &[1u8, 2, 4, 8] {
        let mut params = enc_params(width, height);
        params.pixel_format = Some(PixelFormat::Yuv422P);
        let cfg = EncoderConfig::default().with_mbs_per_slice(m);
        let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder");
        enc.send_frame(&Frame::Video(src.clone()))
            .expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");
        let v = decode_packet(&pkt.data, None).unwrap_or_else(|e| {
            panic!("mbs_per_slice={m}: decode failed: {e}");
        });
        assert_eq!(v.planes.len(), 3, "mbs_per_slice={m}");
        // Luma PSNR must be effectively lossless at the default qi for
        // a flat-quant encode of an 8-bit synthetic source. We compute
        // SSE explicitly and assert a 38 dB floor — 4:2:2 quantisation
        // adds chroma noise but luma should be well above 40 dB.
        let y_src = &src.planes[0].data;
        let y_dec = &v.planes[0].data;
        assert_eq!(y_src.len(), y_dec.len());
        let mut sse: u64 = 0;
        for (&a, &b) in y_src.iter().zip(y_dec.iter()) {
            let d = a as i32 - b as i32;
            sse += (d * d) as u64;
        }
        let mse = sse as f64 / y_src.len() as f64;
        let psnr = if mse > 0.0 {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        } else {
            99.0
        };
        assert!(
            psnr >= 38.0,
            "mbs_per_slice={m}: luma PSNR {psnr:.2} dB < 38 dB floor"
        );
    }
}

/// Packet size is monotonic in the slice count: smaller `mbs_per_slice`
/// means more slices, more per-slice overhead, and a larger packet.
/// The slice fixed overhead is the slice_table u16 entry (2 bytes) +
/// the 6-byte slice_header (no alpha). For a 128×64 frame (8×4 MBs):
///   - `mbs_per_slice=8`: 4 rows × 1 slice  =   4 slices.
///   - `mbs_per_slice=4`: 4 rows × 2 slices =   8 slices.
///   - `mbs_per_slice=2`: 4 rows × 4 slices =  16 slices.
///   - `mbs_per_slice=1`: 4 rows × 8 slices =  32 slices.
///
/// Even though the entropy-coded payloads shrink slightly per slice,
/// the (header + table) overhead dominates and the total grows
/// monotonically. We assert pkt(1) > pkt(2) > pkt(4) > pkt(8).
#[test]
fn packet_size_monotonic_in_slice_count() {
    let width = 128u32;
    let height = 64u32;
    let pkt_8 = encode_at_mbs_per_slice(width, height, Some(8));
    let pkt_4 = encode_at_mbs_per_slice(width, height, Some(4));
    let pkt_2 = encode_at_mbs_per_slice(width, height, Some(2));
    let pkt_1 = encode_at_mbs_per_slice(width, height, Some(1));
    eprintln!(
        "128x64 4:2:2 packet sizes: 8={}B  4={}B  2={}B  1={}B (1/8 ratio = {:.3}×)",
        pkt_8.len(),
        pkt_4.len(),
        pkt_2.len(),
        pkt_1.len(),
        pkt_1.len() as f64 / pkt_8.len() as f64,
    );
    assert!(
        pkt_4.len() > pkt_8.len(),
        "halving mbs_per_slice should grow the packet (4={}B vs 8={}B)",
        pkt_4.len(),
        pkt_8.len()
    );
    assert!(
        pkt_2.len() > pkt_4.len(),
        "halving mbs_per_slice should grow the packet (2={}B vs 4={}B)",
        pkt_2.len(),
        pkt_4.len()
    );
    assert!(
        pkt_1.len() > pkt_2.len(),
        "halving mbs_per_slice should grow the packet (1={}B vs 2={}B)",
        pkt_1.len(),
        pkt_2.len()
    );
}

/// `mbs_per_slice == None` (the default) must produce a packet that's
/// bit-identical to `Some(DEFAULT_MBS_PER_SLICE)` — the knob is purely
/// additive and the default path is unaffected.
#[test]
fn default_path_matches_explicit_eight() {
    let width = 64u32;
    let height = 48u32;
    let pkt_default = encode_at_mbs_per_slice(width, height, None);
    let pkt_explicit = encode_at_mbs_per_slice(width, height, Some(DEFAULT_MBS_PER_SLICE));
    assert_eq!(
        pkt_default,
        pkt_explicit,
        "default mbs_per_slice path must be byte-identical to explicit 8 \
         (default={} B, explicit={} B)",
        pkt_default.len(),
        pkt_explicit.len()
    );
}

/// Builder chain: `with_mbs_per_slice` round-trips through the field.
#[test]
fn with_mbs_per_slice_builder_sets_field() {
    for &m in &[1u8, 2, 4, 8] {
        let cfg = EncoderConfig::default().with_mbs_per_slice(m);
        assert_eq!(cfg.mbs_per_slice, Some(m));
    }
}

/// When `mbs_per_slice == 1` every slice covers a single macroblock,
/// so the picture's slice count == mb_width * mb_height. For an 8×4
/// MB picture that's 32 slices. The decoder must accept this without
/// rejecting (defence-in-depth check that the smallest configuration
/// flows through the entire pipeline cleanly).
#[test]
fn mbs_per_slice_one_yields_max_slice_count() {
    let width = 128u32; // 8 MBs wide
    let height = 64u32; // 4 MBs tall
    let pkt = encode_at_mbs_per_slice(width, height, Some(1));
    let (_, pictures) = parse_frame(&pkt).expect("parse_frame");
    let (ph, _) = parse_picture_header(pictures).expect("parse_picture_header");
    assert_eq!(
        ph.deprecated_number_of_slices, 32,
        "8x4 MBs with mbs_per_slice=1 must emit 32 slices, got {}",
        ph.deprecated_number_of_slices
    );
    // And the decode round-trips.
    let v = decode_packet(&pkt, None).unwrap_or_else(|e| panic!("decode failed: {e}"));
    assert_eq!(v.planes.len(), 3);
}
