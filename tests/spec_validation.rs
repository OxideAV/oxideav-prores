//! Integration tests for RDD 36 syntax-element validation in the decoder.
//!
//! Each test constructs a hand-built frame header that exercises one
//! "decoder shall refuse" clause from the spec and verifies the
//! decoder rejects it. The clauses covered are:
//!
//! * §6.1.1 / §6.4 `bitstream_version`: anything > 1 is unsupported.
//! * §6.4: a version-0 stream MUST have `chroma_format == 4:2:2` AND
//!   `alpha_channel_type == 0`. Any version-0 stream that carries 4:4:4
//!   chroma or any encoded alpha is malformed.
//! * §6.1.1 Table 2 `interlace_mode`: value 3 is reserved.
//! * §6.1.1 `luma_quantization_matrix` / `chroma_quantization_matrix`:
//!   every entry MUST be in 2..=63.
//!
//! These cases cannot be produced by ffmpeg's `prores_ks` (which itself
//! conforms to RDD 36) or by this crate's own encoder, so the fixtures
//! are synthesised in-test.

use oxideav_prores::frame::{
    parse_frame_header, write_frame, write_frame_with_alpha, ChromaFormat,
};

/// Build a minimal 20-byte frame header (no qmats) with caller-overridable
/// fields. The output starts at the frame_header_size field — the
/// 8-byte `frame_size + 'icpf'` prelude is NOT included since
/// `parse_frame_header` consumes only the header itself.
fn build_frame_header(
    bitstream_version: u8,
    chroma_code: u8,
    interlace_mode: u8,
    alpha_channel_type: u8,
) -> Vec<u8> {
    let mut h = Vec::with_capacity(20);
    h.extend_from_slice(&20u16.to_be_bytes()); // frame_header_size = 20
    h.push(0); // reserved
    h.push(bitstream_version);
    h.extend_from_slice(b"Lavc"); // encoder_id
    h.extend_from_slice(&64u16.to_be_bytes()); // width
    h.extend_from_slice(&48u16.to_be_bytes()); // height
                                               // byte 12: chroma_format(2) + reserved(2) + interlace_mode(2) + reserved(2)
    h.push(((chroma_code & 0x3) << 6) | ((interlace_mode & 0x3) << 2));
    h.push(0); // aspect/frame_rate
    h.push(0); // color_primaries
    h.push(0); // transfer
    h.push(0); // matrix
    h.push(alpha_channel_type & 0x0F); // reserved(4) + alpha_channel_type(4)
    h.push(0); // reserved
    h.push(0); // load flags = 0
    debug_assert_eq!(h.len(), 20);
    h
}

#[test]
fn rejects_bitstream_version_above_one() {
    // RDD 36 §6.1.1 / §6.4: "A decoder shall abort if it encounters a
    // bitstream with an unsupported bitstream_version value."
    for v in [2u8, 5, 255] {
        let h = build_frame_header(v, 2, 0, 0);
        let err = parse_frame_header(&h).expect_err("must reject");
        assert!(
            format!("{err:?}").contains("bitstream_version"),
            "error for v={v} mentions bitstream_version: {err:?}"
        );
    }
}

#[test]
fn accepts_bitstream_version_zero_with_422_no_alpha() {
    // The only conforming version-0 stream: 4:2:2, no alpha.
    let h = build_frame_header(0, 2, 0, 0);
    let (fh, _) = parse_frame_header(&h).expect("must accept");
    assert_eq!(fh.bitstream_version, 0);
    assert_eq!(fh.chroma_format, ChromaFormat::Y422);
    assert_eq!(fh.alpha_channel_type, 0);
}

#[test]
fn rejects_version_zero_with_444_chroma() {
    // RDD 36 §6.4: "Version 0 bitstreams will have a value of 2 (4:2:2
    // sampling) for the chroma_format syntax element." A version-0
    // stream that carries chroma_format=3 (4:4:4) is malformed.
    let h = build_frame_header(0, 3, 0, 0);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("bitstream_version 0") && msg.contains("chroma_format"),
        "error mentions both v0 + chroma_format: {msg}"
    );
}

#[test]
fn rejects_version_zero_with_alpha_8bit() {
    // RDD 36 §6.4: "Version 0 bitstreams will have […] a value of 0
    // (no encoded alpha) for the alpha_channel_type element."
    let h = build_frame_header(0, 2, 0, 1);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("bitstream_version 0") && msg.contains("alpha_channel_type"),
        "error mentions both v0 + alpha_channel_type: {msg}"
    );
}

#[test]
fn rejects_version_zero_with_alpha_16bit() {
    let h = build_frame_header(0, 2, 0, 2);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("bitstream_version 0") && msg.contains("alpha_channel_type"),
        "error mentions both v0 + alpha_channel_type: {msg}"
    );
}

#[test]
fn accepts_version_one_with_444_chroma() {
    // 4:4:4 is a v1 feature; version 1 + chroma_format=3 + no alpha is
    // the canonical ap4h stream layout.
    let h = build_frame_header(1, 3, 0, 0);
    let (fh, _) = parse_frame_header(&h).expect("must accept");
    assert_eq!(fh.bitstream_version, 1);
    assert_eq!(fh.chroma_format, ChromaFormat::Y444);
    assert_eq!(fh.alpha_channel_type, 0);
}

#[test]
fn accepts_version_one_with_alpha_on_422() {
    // §6.4 explicitly notes the value of alpha_channel_type can be 0
    // when bitstream_version is 1, but the converse — alpha != 0 when
    // bitstream_version == 1 — is also legal. Tests 4:2:2 + 8-bit alpha.
    let h = build_frame_header(1, 2, 0, 1);
    let (fh, _) = parse_frame_header(&h).expect("must accept");
    assert_eq!(fh.bitstream_version, 1);
    assert_eq!(fh.chroma_format, ChromaFormat::Y422);
    assert_eq!(fh.alpha_channel_type, 1);
}

#[test]
fn rejects_interlace_mode_three_reserved() {
    // RDD 36 §6.1.1 Table 2: interlace_mode==3 is "Reserved".
    let h = build_frame_header(1, 2, 3, 0);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("interlace_mode 3"),
        "error mentions reserved interlace_mode 3: {msg}"
    );
}

#[test]
fn accepts_interlace_mode_zero_one_two() {
    // 0 = progressive, 1 = TFF, 2 = BFF; all conforming.
    for im in [0u8, 1, 2] {
        let h = build_frame_header(1, 2, im, 0);
        let (fh, _) = parse_frame_header(&h).expect("must accept");
        assert_eq!(fh.interlace_mode, im, "interlace_mode {im}");
    }
}

#[test]
fn rejects_luma_qmat_entry_below_two() {
    // §6.1.1: each entry is "in the range 2, 3, …, 63". Build a header
    // that loads a luma matrix with one entry = 1 (below floor).
    let mut h = build_frame_header(1, 2, 0, 0);
    // Patch fh_size to 84 (= 20 + 64) and set load_luma flag.
    h[0..2].copy_from_slice(&84u16.to_be_bytes());
    h[19] = 0b10; // load_luma=1, load_chroma=0
    let mut bad = [4u8; 64];
    bad[17] = 1; // floor violation
    h.extend_from_slice(&bad);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("luma_quantization_matrix") && msg.contains("2..=63"),
        "error mentions luma qmat range: {msg}"
    );
}

#[test]
fn rejects_luma_qmat_entry_above_63() {
    let mut h = build_frame_header(1, 2, 0, 0);
    h[0..2].copy_from_slice(&84u16.to_be_bytes());
    h[19] = 0b10;
    let mut bad = [4u8; 64];
    bad[0] = 64; // ceiling violation
    h.extend_from_slice(&bad);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("luma_quantization_matrix") && msg.contains("2..=63"),
        "error mentions luma qmat range: {msg}"
    );
}

#[test]
fn rejects_chroma_qmat_entry_out_of_range() {
    // load_luma=1, load_chroma=1; valid luma matrix but chroma carries
    // an entry of 0.
    let mut h = build_frame_header(1, 3, 0, 0);
    h[0..2].copy_from_slice(&148u16.to_be_bytes()); // 20 + 64 + 64
    h[19] = 0b11;
    let luma = [4u8; 64];
    h.extend_from_slice(&luma);
    let mut chroma = [4u8; 64];
    chroma[42] = 0;
    h.extend_from_slice(&chroma);
    let err = parse_frame_header(&h).expect_err("must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("chroma_quantization_matrix") && msg.contains("2..=63"),
        "error mentions chroma qmat range: {msg}"
    );
}

#[test]
fn accepts_qmats_with_spec_endpoints() {
    // Build a header carrying a 64-entry matrix whose endpoints are 2
    // and 63 (the inclusive bounds). Must decode the header cleanly.
    let mut h = build_frame_header(1, 2, 0, 0);
    h[0..2].copy_from_slice(&84u16.to_be_bytes());
    h[19] = 0b10;
    let mut q = [4u8; 64];
    q[0] = 2;
    q[63] = 63;
    h.extend_from_slice(&q);
    let (fh, _) = parse_frame_header(&h).expect("must accept");
    assert_eq!(fh.luma_qmat[0], 2);
    assert_eq!(fh.luma_qmat[63], 63);
    // load_chroma=0 with load_luma=1: chroma uses the loaded luma
    // matrix (§6.1.1 load_chroma_quantization_matrix semantics).
    assert_eq!(fh.chroma_qmat, fh.luma_qmat);
}

#[test]
fn encoder_round_trips_v0_compatibility() {
    // The crate's own writer should pick version=0 for the canonical
    // 4:2:2 no-alpha stream, and the decoder must accept it.
    let luma = [4u8; 64];
    let chroma = [4u8; 64];
    let mut buf = Vec::new();
    write_frame(
        &mut buf,
        28,
        128,
        128,
        ChromaFormat::Y422,
        0,
        &luma,
        &chroma,
        false,
        false,
    );
    // Drop the 8-byte container preamble (frame_size + 'icpf').
    let (fh, _) = parse_frame_header(&buf[8..]).expect("must accept own output");
    assert_eq!(
        fh.bitstream_version, 0,
        "encoder must pick v0 for 4:2:2 no-alpha"
    );
}

#[test]
fn encoder_round_trips_v1_compatibility() {
    // 4:4:4 → v1; any-chroma + alpha → v1.
    let luma = [4u8; 64];
    let chroma = [4u8; 64];

    let mut buf444 = Vec::new();
    write_frame(
        &mut buf444,
        28,
        64,
        64,
        ChromaFormat::Y444,
        0,
        &luma,
        &chroma,
        false,
        false,
    );
    let (fh, _) = parse_frame_header(&buf444[8..]).unwrap();
    assert_eq!(fh.bitstream_version, 1, "4:4:4 forces v1");

    let mut buf_alpha = Vec::new();
    write_frame_with_alpha(
        &mut buf_alpha,
        28,
        64,
        64,
        ChromaFormat::Y422,
        0,
        &luma,
        &chroma,
        false,
        false,
        1, // 8-bit alpha
    );
    let (fh, _) = parse_frame_header(&buf_alpha[8..]).unwrap();
    assert_eq!(fh.bitstream_version, 1, "alpha forces v1");
}
