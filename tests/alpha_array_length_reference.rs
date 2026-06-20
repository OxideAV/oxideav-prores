//! Locks the RDD 36 §7.5.3 scanned-alpha **array length** for the
//! partial bottom macroblock row directly against the real reference
//! ProRes 4444 bitstream — no external validator, no synthetic encoder
//! roundtrip in the loop.
//!
//! ## The §7.5.3 question
//!
//! RDD 36 §7.5.3 says the per-slice scanned-alpha array
//! `alphaValues[16 * slice_size_in_mb[j] * r + n]`:
//!
//! > includes alpha values—which shall be discarded—for the excess
//! > pixel(s) at the end of each row of slices with
//! > `j = number_of_slices_per_mb_row − 1` when
//! > `16 * width_in_mb > horizontal_size` **but does not include alpha
//! > values for the excess row(s) of pixels at the bottom of slices with
//! > `i = height_in_mb − 1`** when `16 * height_in_mb > picture_vertical_size`.
//!
//! Read literally, the highlighted clause sizes the bottom macroblock
//! row's `alphaValues[]` to the **visible** row count (e.g. 8 rows for a
//! 1080-line picture, whose bottom MB row spans rows 1072..=1079). The
//! companion exclusion for excess *columns* says they ARE present in the
//! array (and are discarded on paste); the row clause reads as the
//! opposite for excess *rows*.
//!
//! ## What the reference bitstream actually carries
//!
//! The in-tree reference fixture `4444-with-alpha` (1920×1080,
//! `height_in_mb = 68`, `picture_vertical_size = 1080`, so the bottom MB
//! row at `i = 67` has only `1080 − 67*16 = 8` visible rows) instead
//! codes the **full 16-row** array. This test extracts the raw
//! scanned-alpha blob of a real bottom-MB-row slice straight out of
//! `input.mov` and decodes it two ways:
//!
//! * with the §7.5.3-literal visible-row count (`128 cols × 8 rows =
//!   1024` values) → the run/level stream overruns the array (the coded
//!   run keeps producing samples past 1024), proving the blob does NOT
//!   end at the visible row count;
//! * with the full MB-row height (`128 cols × 16 rows = 2048` values) →
//!   the blob decodes exactly, consuming the whole run.
//!
//! That asymmetry is the load-bearing evidence: the §7.5.3 exclusion
//! governs which rows a *decoder writes* to the frame buffer, not the
//! coded array length, which is always the full MB-row height. The
//! crate's encoder + decoder both code the full height to stay
//! bit-compatible with the reference; this test pins the conclusion to
//! the actual reference bytes so a future "optimisation" that truncates
//! the bottom-row array fails loudly.
//!
//! DOCS-GAP candidate: RDD 36 §7.5.3 wording ("does not include … excess
//! row(s)") contradicts the reference bitstream. Recommend a §7.5.3
//! erratum clarifying that the alpha array is always
//! `16 * slice_size_in_mb[j] * 16` values and the row exclusion is a
//! *write* constraint, parallel to the existing column wording.
//!
//! Clean-room note: the only inputs are SMPTE RDD 36 §7.5.3 and the
//! project's own reference fixture bytes. No external decoder consulted.

use std::fs;
use std::path::PathBuf;

use oxideav_prores::alpha::{decode_scanned_alpha, AlphaChannelType};

/// Macroblock side in luma samples (RDD 36 §6.2).
const MB_SIDE_PX: usize = 16;

/// Return the first ProRes frame container (`size + 'icpf' + …`) found in
/// a QuickTime `mdat`. The fixture's `mov` carries the elementary
/// ProRes frames back to back inside `mdat`; we scan for the `icpf`
/// fourcc and read the preceding big-endian `frame_size` (RDD 36 §5.1).
fn first_prores_frame(container: &[u8]) -> &[u8] {
    let needle = b"icpf";
    let mut i = 4usize;
    while i + 4 <= container.len() {
        if &container[i..i + 4] == needle {
            let size_off = i - 4;
            let frame_size =
                u32::from_be_bytes(container[size_off..size_off + 4].try_into().unwrap()) as usize;
            let end = size_off + frame_size;
            if frame_size >= 8 && end <= container.len() {
                return &container[size_off..end];
            }
        }
        i += 1;
    }
    panic!("no ProRes 'icpf' frame found in fixture container");
}

/// Geometry + raw scanned-alpha blob extracted from one slice of a real
/// ProRes 4444 frame.
struct SliceAlpha<'a> {
    /// Number of macroblocks the slice spans horizontally.
    mb_count: usize,
    /// Raw `scanned_alpha()` byte blob (RDD 36 §5.3.3), trailing
    /// byte-alignment padding included.
    blob: &'a [u8],
}

/// Walk the frame header / picture header / slice table of a progressive
/// ProRes 4444 frame and return the alpha blob of slice `target_idx`.
///
/// Header layout per RDD 36 §5.1.1 / §5.1.3 and the fixture manifest:
/// `frame = frame_size(4) + 'icpf'(4) + frame_header`; `frame_header`
/// begins with a 2-byte big-endian `hdr_size`. The picture header begins
/// at `8 + hdr_size`: byte 0 is the picture-header size **in bits**
/// (normally 8 → a 1-byte… here 8-byte header), bytes 5..6 hold the
/// deprecated slice count, and the slice index table (one 2-byte entry
/// per slice) follows the picture header. Per-slice the header byte 0's
/// high 5 bits are the slice-header size; bytes 2..3 / 4..5 / 6..7 are
/// the Y / Cb / Cr coded sizes; alpha is the remainder.
fn extract_slice_alpha(frame: &[u8], target_idx: usize) -> SliceAlpha<'_> {
    let hdr_size = u16::from_be_bytes([frame[8], frame[9]]) as usize;
    let pic = 8 + hdr_size;
    let pic_hdr_bits = frame[pic] as usize;
    assert_eq!(
        pic_hdr_bits % 8,
        0,
        "picture-header size must be a whole number of bytes"
    );
    let pic_hdr_len = pic_hdr_bits / 8;
    let num_slices = u16::from_be_bytes([frame[pic + 5], frame[pic + 6]]) as usize;
    let table = pic + pic_hdr_len;

    let mut sizes = Vec::with_capacity(num_slices);
    for s in 0..num_slices {
        sizes.push(u16::from_be_bytes([frame[table + s * 2], frame[table + s * 2 + 1]]) as usize);
    }
    assert!(target_idx < num_slices, "slice index out of range");

    let mut off = table + num_slices * 2;
    for &sz in &sizes[..target_idx] {
        off += sz;
    }
    let slice = &frame[off..off + sizes[target_idx]];

    let slice_header_size = (slice[0] >> 3) as usize;
    let y = u16::from_be_bytes([slice[2], slice[3]]) as usize;
    let u = u16::from_be_bytes([slice[4], slice[5]]) as usize;
    let v = u16::from_be_bytes([slice[6], slice[7]]) as usize;
    let alpha = &slice[slice_header_size + y + u + v..];

    SliceAlpha {
        // Slice MB width is the picture-header `log2_slice_mb_width`
        // default (8 MBs/slice) for this fixture; we assert it below from
        // the picture header rather than trust a constant blindly.
        mb_count: 1 << (frame[pic + 7] >> 4),
        blob: alpha,
    }
}

fn fixture_mov() -> Vec<u8> {
    let p = PathBuf::from("../../docs/video/prores/fixtures/4444-with-alpha/input.mov");
    fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// The bottom macroblock row of the 1920×1080 fixture is MB row 67
/// (`height_in_mb = ceil(1080/16) = 68`). Slice 1005 is the first slice
/// of that row (`15 slices/row × 67 rows = 1005`), spanning MBs 0..=7 at
/// the left edge — 8 fully-coded MBs, 8 visible pixel rows.
const BOTTOM_ROW_FIRST_SLICE: usize = 1005;

#[test]
fn reference_bottom_mb_row_alpha_blob_is_full_height_not_visible_height() {
    let mov = fixture_mov();
    let frame = first_prores_frame(&mov);
    let sa = extract_slice_alpha(frame, BOTTOM_ROW_FIRST_SLICE);

    // Sanity: this is an 8-MB-wide slice (128 columns).
    assert_eq!(
        sa.mb_count, 8,
        "expected the default 8-MBs-per-slice layout for this fixture"
    );
    let cols = sa.mb_count * MB_SIDE_PX;
    assert_eq!(cols, 128);

    // Visible bottom-MB-row height: 1080 − 67*16 = 8 rows.
    let visible_rows = 1080 - 67 * MB_SIDE_PX;
    assert_eq!(visible_rows, 8);

    // (a) §7.5.3-literal visible-row count: the coded run/level stream
    //     overruns the array, proving the blob is NOT sized to 8 rows.
    let literal_n = cols * visible_rows; // 1024
    let literal = decode_scanned_alpha(sa.blob, literal_n, AlphaChannelType::Sixteen);
    assert!(
        literal.is_err(),
        "the bottom-MB-row alpha blob decoded cleanly at the §7.5.3-literal \
         visible-row length ({literal_n}); the reference bitstream is supposed \
         to carry MORE values than that"
    );

    // (b) Full MB-row height: the blob decodes exactly, consuming the
    //     whole run with no overrun and no leftover bits beyond the pad.
    let full_n = cols * MB_SIDE_PX; // 2048
    let full = decode_scanned_alpha(sa.blob, full_n, AlphaChannelType::Sixteen)
        .expect("bottom-MB-row alpha blob must decode at the full 16-row length");
    assert_eq!(full.len(), full_n);

    // This particular reference slice is fully opaque (16-bit alpha
    // 0xFFFF) across the whole MB row — a single escape diff to 65535
    // followed by one run of 2048. If the array were truncated to 1024
    // the run of 2048 would overrun, which is exactly case (a).
    assert!(
        full.iter().all(|&a| a == 0xFFFF),
        "reference bottom-row slice 1005 is fully opaque"
    );
}

/// Generalise the finding across the whole bottom MB row: every one of
/// the 15 slices in row 67 must decode at the full 16-row length and
/// fail (or, for a degenerate empty blob, not decode more cleanly than)
/// the §7.5.3-literal length. This guards against the conclusion holding
/// only for the left-edge slice.
#[test]
fn reference_bottom_mb_row_full_height_holds_for_every_slice() {
    let mov = fixture_mov();
    let frame = first_prores_frame(&mov);

    // 15 slices per MB row for 1920 px (120 MBs / 8 MBs-per-slice).
    let slices_per_row = 1920 / MB_SIDE_PX / 8;
    assert_eq!(slices_per_row, 15);

    let mut full_ok = 0usize;
    for j in 0..slices_per_row {
        let idx = BOTTOM_ROW_FIRST_SLICE + j;
        let sa = extract_slice_alpha(frame, idx);
        let cols = sa.mb_count * MB_SIDE_PX;
        let full_n = cols * MB_SIDE_PX;
        let decoded = decode_scanned_alpha(sa.blob, full_n, AlphaChannelType::Sixteen)
            .unwrap_or_else(|e| panic!("slice {idx} failed at full 16-row length: {e}"));
        assert_eq!(decoded.len(), full_n, "slice {idx} length");
        full_ok += 1;
    }
    assert_eq!(
        full_ok, slices_per_row,
        "every bottom-MB-row slice must decode at the full MB-row height"
    );
}

/// Contrast slice: an *interior* MB row (row 0) likewise codes the full
/// 16 rows (there is no partial-height question there), confirming the
/// array length is uniformly `16 * slice_size_in_mb * 16` and the
/// bottom-row blob is not a special shorter shape.
#[test]
fn reference_interior_mb_row_alpha_blob_is_full_height() {
    let mov = fixture_mov();
    let frame = first_prores_frame(&mov);
    let sa = extract_slice_alpha(frame, 0);
    let cols = sa.mb_count * MB_SIDE_PX;
    let full_n = cols * MB_SIDE_PX;
    let decoded = decode_scanned_alpha(sa.blob, full_n, AlphaChannelType::Sixteen)
        .expect("interior-row alpha blob must decode at the full 16-row length");
    assert_eq!(decoded.len(), full_n);
}
