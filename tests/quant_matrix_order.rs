//! Order-pinned regression tests for quantisation-matrix storage: the
//! ProRes frame header carries the 8x8 weight matrices in **natural
//! (raster) coefficient order**, row-major — NOT in the block-scan
//! order used to read coefficients inside a slice.
//!
//! The in-tree corpus documentation originally described the tables as
//! "zigzag order, permute before use"; that wording was an erratum,
//! corrected in the corpus (`docs/video/prores/
//! prores-fixtures-and-traces.md`, Errata E1). This crate has always
//! applied the matrices in natural order — this suite pins that fact so
//! a regression that introduces a scan permutation anywhere (header
//! write, header parse, or dequantisation) fails loudly:
//!
//! 1. **Write side, raw bytes** — the encoder places the configured
//!    matrix on the wire verbatim at the fixed §6.1.1 header offsets
//!    (no permutation), checked against the byte buffer directly, with
//!    no parser in the loop.
//! 2. **Raster fingerprints** — the wire bytes of the non-flat
//!    signature tables reshape row-major into a 2-D-monotone matrix
//!    (weights non-decreasing along every row and column), and the
//!    Proxy 63-clamp forms a closed bottom-right triangle. Both
//!    fingerprints *fail* when the same bytes are reinterpreted as
//!    scan-ordered, so the tests genuinely discriminate the two orders.
//! 3. **Reference fixture bytes** — the corpus streams carry the same
//!    natural-order bytes at the same offsets (read raw, no parser).
//! 4. **Dequantisation indexing** — an independent in-test
//!    reconstruction (shared leaf primitives only: entropy/slice
//!    decode, IDCT, scan tables) that scales the coefficient at natural
//!    position `k` by `qmat[k]` reproduces the production decoder's
//!    output byte-for-byte, while the scan-permuted alternative does
//!    not. Equivalently: the coefficient read at scanned index `s`
//!    (natural position `n = INV_SCAN[s]`) is scaled by `qmat[n]`.
//!
//! Validator-independent; the fixture check skips when the `docs/`
//! corpus is absent (standalone CI), matching `quant_matrix_signature.rs`.

use std::fs;
use std::path::PathBuf;

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::dct::{idct8x8, idct8x8_dc_only, is_dc_only};
use oxideav_prores::decoder::{decode_packet, BitDepth};
use oxideav_prores::encoder::encode_frame_with_qmats;
use oxideav_prores::frame::{
    compute_slice_sizes, parse_frame, parse_picture_header, parse_slice_header, ChromaFormat,
    Profile,
};
use oxideav_prores::quant::{
    qscale, QuantMatrices, BLOCK_SCAN_PROGRESSIVE, SIGNATURE_LT_QMAT, SIGNATURE_PROXY_CHROMA_QMAT,
    SIGNATURE_PROXY_LUMA_QMAT, SIGNATURE_STANDARD_QMAT,
};
use oxideav_prores::slice::decode_slice_components;

// Absolute byte offsets within a ProRes frame unit: frame_size u32 at
// [0..4], 'icpf' at [4..8], frame_header_size u16 at [8..10]; the frame
// header body starts at offset 8, so the load-flags byte
// (header-relative offset 19) is at 27 and the first quantisation table
// (header-relative 20) begins at 28. A second table, when present,
// follows at 92.
const FH_SIZE_OFF: usize = 8;
const FLAGS_OFF: usize = 27;
const LUMA_TABLE: std::ops::Range<usize> = 28..92;
const CHROMA_TABLE: std::ops::Range<usize> = 92..156;

/// True when the row-major 8x8 reshape of `m` is non-decreasing along
/// every row (left→right, increasing horizontal frequency `u`) and every
/// column (top→bottom, increasing vertical frequency `v`) — the 2-D
/// gradient a quantisation matrix has only in natural raster order.
fn is_2d_monotone(m: &[u8; 64]) -> bool {
    for v in 0..8 {
        for u in 0..7 {
            if m[v * 8 + u] > m[v * 8 + u + 1] {
                return false;
            }
        }
    }
    for u in 0..8 {
        for v in 0..7 {
            if m[v * 8 + u] > m[(v + 1) * 8 + u] {
                return false;
            }
        }
    }
    true
}

/// Reinterpret wire bytes under the rejected hypothesis "the table is
/// stored in block-scan order": if `wire` were scan-ordered, the weight
/// for natural position `n` would live at wire index
/// `BLOCK_SCAN_PROGRESSIVE[n]`.
fn as_if_scan_ordered(wire: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for n in 0..64 {
        out[n] = wire[BLOCK_SCAN_PROGRESSIVE[n] as usize];
    }
    out
}

/// True when the set of 63-valued entries is closed under moving right
/// and down in the row-major 8x8 reshape — the bottom-right triangle a
/// high-frequency clamp forms only in natural raster order.
fn clamp63_is_closed_triangle(m: &[u8; 64]) -> bool {
    for v in 0..8 {
        for u in 0..8 {
            if m[v * 8 + u] == 63 {
                if u < 7 && m[v * 8 + u + 1] != 63 {
                    return false;
                }
                if v < 7 && m[(v + 1) * 8 + u] != 63 {
                    return false;
                }
            }
        }
    }
    true
}

fn synth_422(w: usize, h: usize) -> VideoFrame {
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            // Detail-rich pattern so quantised AC coefficients survive.
            y[j * w + i] = (((i * 13 + j * 7) as u8) ^ ((i * j) as u8)).wrapping_add(24);
        }
        for i in 0..cw {
            cb[j * cw + i] = 96 + (((i * 5) ^ (j * 3)) as u8 & 0x3F);
            cr[j * cw + i] = (150u8).wrapping_sub(((i * 3 + j * 5) as u8) & 0x3F);
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

// -------------------------------------------------------------------
// 1. Write side: raw wire bytes, no parser in the loop.
// -------------------------------------------------------------------

#[test]
fn encoder_places_matrix_bytes_on_wire_verbatim_in_natural_order() {
    // Standard signature: minimal carriage (1, 0) — one 64-byte table.
    let pkt = encode_frame_with_qmats(
        &synth_422(32, 32),
        32,
        32,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        QuantMatrices::signature_for_profile(Profile::Standard),
    )
    .expect("encode standard");
    let fh_size = u16::from_be_bytes([pkt[FH_SIZE_OFF], pkt[FH_SIZE_OFF + 1]]);
    assert_eq!(fh_size, 84, "one carried table → 84-byte frame header");
    assert_eq!(pkt[FLAGS_OFF] & 0b11, 0b10, "flags (1, 0)");
    assert_eq!(
        &pkt[LUMA_TABLE],
        &SIGNATURE_STANDARD_QMAT[..],
        "luma table must appear on the wire verbatim (natural raster order)"
    );

    // Proxy signature: both tables (1, 1) — luma then chroma, each verbatim.
    let pkt = encode_frame_with_qmats(
        &synth_422(32, 32),
        32,
        32,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Proxy,
        8,
        QuantMatrices::signature_for_profile(Profile::Proxy),
    )
    .expect("encode proxy");
    let fh_size = u16::from_be_bytes([pkt[FH_SIZE_OFF], pkt[FH_SIZE_OFF + 1]]);
    assert_eq!(fh_size, 148, "two carried tables → 148-byte frame header");
    assert_eq!(pkt[FLAGS_OFF] & 0b11, 0b11, "flags (1, 1)");
    assert_eq!(&pkt[LUMA_TABLE], &SIGNATURE_PROXY_LUMA_QMAT[..]);
    assert_eq!(&pkt[CHROMA_TABLE], &SIGNATURE_PROXY_CHROMA_QMAT[..]);
}

// -------------------------------------------------------------------
// 2. Raster fingerprints discriminate natural from scan order.
// -------------------------------------------------------------------

#[test]
fn natural_order_fingerprints_hold_and_fail_under_scan_reinterpretation() {
    // The non-flat signature tables have the 2-D low→high gradient in
    // natural order…
    for (name, m) in [
        ("proxy luma", &SIGNATURE_PROXY_LUMA_QMAT),
        ("proxy chroma", &SIGNATURE_PROXY_CHROMA_QMAT),
        ("LT", &SIGNATURE_LT_QMAT),
        ("standard", &SIGNATURE_STANDARD_QMAT),
    ] {
        assert!(is_2d_monotone(m), "{name}: natural order must be monotone");
        // …and lose it when the same bytes are reinterpreted as
        // scan-ordered — proving the fingerprint tells the orders apart.
        assert!(
            !is_2d_monotone(&as_if_scan_ordered(m)),
            "{name}: scan reinterpretation must scatter the gradient"
        );
    }

    // Proxy's 63-clamp fills a closed bottom-right triangle in natural
    // order; a scan-ordered reshape breaks the closure.
    for (name, m) in [
        ("proxy luma", &SIGNATURE_PROXY_LUMA_QMAT),
        ("proxy chroma", &SIGNATURE_PROXY_CHROMA_QMAT),
    ] {
        assert!(
            clamp63_is_closed_triangle(m),
            "{name}: 63-clamp must be a closed right/down triangle"
        );
        assert!(
            !clamp63_is_closed_triangle(&as_if_scan_ordered(m)),
            "{name}: scan reinterpretation must break the triangle"
        );
    }
}

// -------------------------------------------------------------------
// 3. Reference fixture bytes carry the natural order on the wire.
// -------------------------------------------------------------------

/// Extract one raw ProRes frame from a fixture container, or `None`
/// when the `docs/` corpus is not checked out (standalone CI).
fn fixture_frame(name: &str) -> Option<Vec<u8>> {
    let path = PathBuf::from("../../docs/video/prores/fixtures")
        .join(name)
        .join("input.mov");
    let container = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {name}: missing {} ({e}). docs/ corpus lives in the \
                 workspace umbrella; the standalone checkout has no fixtures.",
                path.display()
            );
            return None;
        }
    };
    let needle = b"icpf";
    let mut i = 4usize;
    while i + 4 <= container.len() {
        if &container[i..i + 4] == needle {
            let size_off = i - 4;
            let frame_size =
                u32::from_be_bytes(container[size_off..size_off + 4].try_into().unwrap()) as usize;
            let end = size_off + frame_size;
            if end <= container.len() && frame_size >= 8 {
                return Some(container[size_off..end].to_vec());
            }
        }
        i += 1;
    }
    panic!("{name}: no ProRes frame found in fixture container");
}

#[test]
fn fixture_wire_bytes_are_natural_order_at_fixed_offsets() {
    // (fixture, expected luma table, expected chroma table). Every
    // corpus stream carries both tables explicitly (flags 0b11).
    let cases: [(&str, &[u8; 64], &[u8; 64]); 3] = [
        (
            "sq-1920x1080",
            &SIGNATURE_STANDARD_QMAT,
            &SIGNATURE_STANDARD_QMAT,
        ),
        ("lt-1280x720", &SIGNATURE_LT_QMAT, &SIGNATURE_LT_QMAT),
        (
            "proxy-1280x720",
            &SIGNATURE_PROXY_LUMA_QMAT,
            &SIGNATURE_PROXY_CHROMA_QMAT,
        ),
    ];
    for (name, want_luma, want_chroma) in cases {
        let Some(frame) = fixture_frame(name) else {
            return;
        };
        // Raw bytes at fixed offsets — deliberately no parse_frame call,
        // so a parser-side permutation could not mask a wire-order change.
        assert_eq!(frame[FLAGS_OFF] & 0b11, 0b11, "{name}: flags (1, 1)");
        assert_eq!(
            &frame[LUMA_TABLE],
            &want_luma[..],
            "{name}: luma wire bytes"
        );
        assert_eq!(
            &frame[CHROMA_TABLE],
            &want_chroma[..],
            "{name}: chroma wire bytes"
        );
        // The reference bytes themselves show the natural-order
        // fingerprint (and lose it under scan reinterpretation), so the
        // wire order is pinned by structure as well as by value.
        let wire_luma: [u8; 64] = frame[LUMA_TABLE].try_into().unwrap();
        assert!(is_2d_monotone(&wire_luma), "{name}: wire gradient");
        assert!(
            !is_2d_monotone(&as_if_scan_ordered(&wire_luma)),
            "{name}: scan reinterpretation must fail"
        );
    }
}

// -------------------------------------------------------------------
// 4. Dequantisation indexes the matrix by natural position.
// -------------------------------------------------------------------

/// Splice two 64-byte quantisation tables into a flat-carriage frame
/// (flags (0,0), 20-byte header), producing the explicit (1,1) form.
/// The picture() payload is offset-self-relative, so the splice is
/// well-formed.
fn splice_both_tables(pkt: &[u8], luma: &[u8; 64], chroma: &[u8; 64]) -> Vec<u8> {
    let mut v = pkt.to_vec();
    assert_eq!(
        u16::from_be_bytes([v[FH_SIZE_OFF], v[FH_SIZE_OFF + 1]]),
        20,
        "expected a no-tables frame header to splice into"
    );
    v[FLAGS_OFF] |= 0b11;
    let mut insert = Vec::with_capacity(128);
    insert.extend_from_slice(luma);
    insert.extend_from_slice(chroma);
    v.splice(LUMA_TABLE.start..LUMA_TABLE.start, insert);
    let fh = u16::from_be_bytes([v[FH_SIZE_OFF], v[FH_SIZE_OFF + 1]]) + 128;
    v[FH_SIZE_OFF..FH_SIZE_OFF + 2].copy_from_slice(&fh.to_be_bytes());
    let fs = u32::from_be_bytes([v[0], v[1], v[2], v[3]]) + 128;
    v[0..4].copy_from_slice(&fs.to_be_bytes());
    v
}

/// Independent §7.3 + §7.4 + §7.5.1 reconstruction of a progressive
/// 4:2:2 8-bit single-picture frame, sharing only leaf primitives with
/// the production decoder (entropy/slice decode, IDCT, Table 15). The
/// order-critical step — scaling the coefficient at natural position
/// `k` by `qmat[k]` — is written out here as the pinned ground truth.
/// `luma_w` / `chroma_w` are the weight tables to apply; passing
/// scan-permuted tables models the rejected zigzag hypothesis.
fn reference_reconstruct(
    pkt: &[u8],
    width: usize,
    height: usize,
    luma_w: &[u8; 64],
    chroma_w: &[u8; 64],
) -> Vec<Vec<u8>> {
    let (fh, after_fh) = parse_frame(pkt).expect("parse_frame");
    assert_eq!(fh.picture_count(), 1, "progressive frame expected");
    let (ph, after_ph) = parse_picture_header(after_fh).expect("parse_picture_header");

    let mbs_x = width.div_ceil(16);
    let mbs_y = height.div_ceil(16);
    let template = compute_slice_sizes(mbs_x, ph.log2_desired_slice_size_in_mb);
    let slice_count = template.len() * mbs_y;
    let mut slice_sizes = Vec::with_capacity(slice_count);
    for i in 0..slice_count {
        slice_sizes.push(u16::from_be_bytes([after_ph[i * 2], after_ph[i * 2 + 1]]) as usize);
    }
    let mut cursor = &after_ph[slice_count * 2..];

    let y_stride = mbs_x * 16;
    let c_stride = mbs_x * 8;
    let mut y_plane = vec![0u8; y_stride * mbs_y * 16];
    let mut cb_plane = vec![0u8; c_stride * mbs_y * 16];
    let mut cr_plane = vec![0u8; c_stride * mbs_y * 16];

    // §7.5.1 for b = 8, full range: s = clamp(round((v + 256) / 2), 0, 255).
    let to_sample = |v: f32| -> u8 {
        let s = (v + 256.0) * 0.5;
        if s <= 0.0 {
            0
        } else if s >= 255.0 {
            255
        } else {
            s.round() as u8
        }
    };
    // §7.4 IDCT with the same DC-only specialisation the decoder uses.
    let idct = |blk: &mut [f32; 64]| {
        if is_dc_only(blk) {
            idct8x8_dc_only(blk);
        } else {
            idct8x8(blk);
        }
    };

    let mut slice_idx = 0usize;
    for my in 0..mbs_y {
        let mut mx = 0usize;
        for &tmpl in &template {
            let mb_count = tmpl.min(mbs_x - mx);
            if mb_count == 0 {
                break;
            }
            let coded = slice_sizes[slice_idx];
            slice_idx += 1;
            let slice_data = &cursor[..coded];
            cursor = &cursor[coded..];
            let (sh, after_sh) = parse_slice_header(slice_data, false).expect("slice header");
            let y_len = sh.coded_size_of_y_data as usize;
            let cb_len = sh.coded_size_of_cb_data as usize;
            let cr_len = slice_data.len() - sh.slice_header_size as usize - y_len - cb_len;
            let blocks = decode_slice_components(
                &after_sh[..y_len],
                &after_sh[y_len..y_len + cb_len],
                &after_sh[y_len + cb_len..y_len + cb_len + cr_len],
                mb_count,
                ChromaFormat::Y422,
                false,
            )
            .expect("slice components");

            let qs = qscale(sh.quantization_index) as f32;
            // THE ORDER PIN: `blocks` holds coefficients at natural
            // (raster) positions; the weight for natural position k is
            // `w[k]` — the wire table indexed directly, no permutation.
            // (The coefficient read at scanned index s sits at natural
            // position n with BLOCK_SCAN_PROGRESSIVE[n] == s, and is
            // scaled by w[n].)
            let dequant = |blk: &[i32; 64], w: &[u8; 64]| -> [f32; 64] {
                let mut out = [0.0f32; 64];
                for k in 0..64 {
                    out[k] = (blk[k] as f32 * w[k] as f32 * qs) / 8.0;
                }
                out
            };

            for mb in 0..mb_count {
                let base = mb * 8; // 4 luma + 2 cb + 2 cr at 4:2:2
                let mb_x = mx + mb;
                // Luma blocks at (bx, by) in units of 8 samples.
                for (i, (bx, by)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
                    let mut f = dequant(&blocks[base + i], luma_w);
                    idct(&mut f);
                    for j in 0..8 {
                        for i2 in 0..8 {
                            let x = mb_x * 16 + bx * 8 + i2;
                            let y = my * 16 + by * 8 + j;
                            y_plane[y * y_stride + x] = to_sample(f[j * 8 + i2]);
                        }
                    }
                }
                // Chroma blocks: half-width MB column, two vertical blocks.
                for (i, by) in [0usize, 1].iter().enumerate() {
                    let mut f = dequant(&blocks[base + 4 + i], chroma_w);
                    idct(&mut f);
                    for j in 0..8 {
                        for i2 in 0..8 {
                            let x = mb_x * 8 + i2;
                            let y = my * 16 + by * 8 + j;
                            cb_plane[y * c_stride + x] = to_sample(f[j * 8 + i2]);
                        }
                    }
                }
                for (i, by) in [0usize, 1].iter().enumerate() {
                    let mut f = dequant(&blocks[base + 6 + i], chroma_w);
                    idct(&mut f);
                    for j in 0..8 {
                        for i2 in 0..8 {
                            let x = mb_x * 8 + i2;
                            let y = my * 16 + by * 8 + j;
                            cr_plane[y * c_stride + x] = to_sample(f[j * 8 + i2]);
                        }
                    }
                }
            }
            mx += mb_count;
        }
    }
    vec![y_plane, cb_plane, cr_plane]
}

#[test]
fn dequant_scales_natural_position_k_by_qmat_k() {
    const W: usize = 32;
    const H: usize = 32;
    // Encode with flat matrices (orderless: all 4s), then splice two
    // distinct, strongly non-flat tables into the header so only the
    // DECODE side exercises the order under test.
    let flat_pkt = encode_frame_with_qmats(
        &synth_422(W, H),
        W as u32,
        H as u32,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        2,
        QuantMatrices::flat(),
    )
    .expect("encode flat");
    let pkt = splice_both_tables(&flat_pkt, &SIGNATURE_LT_QMAT, &SIGNATURE_STANDARD_QMAT);

    // Read-side verbatim pin: the parsed header exposes exactly the
    // spliced wire bytes.
    let (fh, _) = parse_frame(&pkt).expect("parse spliced");
    assert_eq!(fh.luma_qmat, SIGNATURE_LT_QMAT);
    assert_eq!(fh.chroma_qmat, SIGNATURE_STANDARD_QMAT);

    let decoded = decode_packet(&pkt, None).expect("decode spliced");
    assert_eq!(decoded.planes.len(), 3);

    // Natural-order reference reproduces the decoder byte-for-byte.
    let natural = reference_reconstruct(&pkt, W, H, &SIGNATURE_LT_QMAT, &SIGNATURE_STANDARD_QMAT);
    for (p, (got, want)) in decoded.planes.iter().zip(natural.iter()).enumerate() {
        assert_eq!(
            got.data, *want,
            "plane {p}: decoder must match the natural-order reference exactly"
        );
    }

    // The scan-permuted alternative (the rejected zigzag hypothesis)
    // must NOT match — the pin has teeth.
    let permuted = reference_reconstruct(
        &pkt,
        W,
        H,
        &as_if_scan_ordered(&SIGNATURE_LT_QMAT),
        &as_if_scan_ordered(&SIGNATURE_STANDARD_QMAT),
    );
    let any_diff = decoded
        .planes
        .iter()
        .zip(permuted.iter())
        .any(|(got, alt)| got.data != *alt);
    assert!(
        any_diff,
        "scan-permuted weights must change the reconstruction — otherwise \
         this test could not detect an order regression"
    );
}
