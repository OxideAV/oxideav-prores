//! Locks the FULL decoder alpha-plane output (RDD 36 §7.5.2 promotion +
//! §7.5.3 placement) byte-for-byte against an independent reconstruction
//! built straight from the reference ProRes 4444 bitstream.
//!
//! ## Why this test exists
//!
//! The `4444-with-alpha` corpus fixture is scored at `DecodesCleanly` in
//! `docs_corpus.rs` because its Y/Cb/Cr planes are gated on the
//! float-vs-fixed IDCT choice that RDD 36 §7.4 leaves to the decoder — no
//! IDCT path is bit-exact against the reference encoder, so those planes
//! can only be PSNR-scored. The **alpha** plane is different: it is coded
//! losslessly (§7.1.2 run/level, no DCT) and promoted to the output bit
//! depth by the deterministic §7.5.2 rule
//! `alphaSample = round((2^b − 1) * alpha ÷ mask)`. There is therefore a
//! single correct alpha plane for any conforming decoder, and this test
//! pins ours to it.
//!
//! ## What is reconstructed independently
//!
//! Rather than trust the decoder against itself, this test walks the
//! frame / picture / slice tables of the real `input.mov` and, for every
//! slice, extracts the raw `scanned_alpha()` blob, decodes it via the
//! public `alpha::decode_scanned_alpha`, applies the §7.5.2 16-bit→12-bit
//! promotion, and places the result into a full-frame plane honouring the
//! §7.5.3 row/column geometry (excess right-edge columns and excess
//! bottom rows discarded). That reconstruction uses ONLY the spec rule
//! and the bitstream bytes; it shares no placement code with the decoder.
//! The decoder's emitted plane 3 must then match it exactly.
//!
//! The §7.5.3 partial-bottom-MB-row subtlety (the coded array is the full
//! 16-row MB height even though only 8 rows of the 1080-line picture are
//! visible — see `alpha_array_length_reference.rs`) is exercised on the
//! real geometry here: a placement bug that wrote the wrong 8 of the 16
//! decoded rows, or that mis-cropped the right edge (1920 is MB-aligned
//! so there is no right-edge crop here, but the column-stride arithmetic
//! still has to land each row at the correct offset), would diverge.
//!
//! Clean-room note: inputs are SMPTE RDD 36 §7.5.2 / §7.5.3 and the
//! project's own reference fixture bytes. No external decoder consulted.

use std::fs;
use std::path::PathBuf;

use oxideav_prores::alpha::{decode_scanned_alpha, AlphaChannelType};
use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

/// Macroblock side in luma samples (RDD 36 §6.2).
const MB_SIDE_PX: usize = 16;

/// 12-bit output: `alphaSample = round((2^12 − 1) * alpha ÷ 65535)`
/// (§7.5.2, 16-bit decoded alpha → 12-bit pixel sample).
fn promote_16_to_12(alpha: u16) -> u16 {
    let max_out = 4095u64;
    let mask = 65535u64;
    let num = max_out * alpha as u64;
    ((num + mask / 2) / mask) as u16
}

/// Return the first ProRes frame container (`size + 'icpf' + …`) in a
/// QuickTime `mdat`. The fixture carries elementary ProRes frames back to
/// back; scan for the `icpf` fourcc and read the preceding big-endian
/// `frame_size` (RDD 36 §5.1).
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

/// Picture / slice geometry parsed from a progressive ProRes 4444 frame
/// header + picture header, plus the byte offset of the first slice's
/// data.
struct Picture {
    /// MBs per slice for the (uniform, full-width) slices of this picture.
    mbs_per_slice: usize,
    /// Total slice count.
    num_slices: usize,
    /// Per-slice coded byte sizes (slice index table).
    slice_sizes: Vec<usize>,
    /// Offset of the first slice's data (just past the slice index table).
    first_slice_off: usize,
    /// Picture width / height in luma samples (from the frame header).
    width: usize,
    height: usize,
}

/// Parse the frame + picture headers of a *progressive* ProRes 4444
/// frame. Layout per RDD 36 §5.1 and the fixture manifest:
/// `frame = frame_size(4) + 'icpf'(4) + frame_header`. The frame header
/// begins with `hdr_size`(2 BE); width/height live at header bytes 8/10.
/// The picture header begins at `8 + hdr_size`: byte 0 is the picture
/// header size in BITS (normally 8 → 1 byte… here the table treats it as
/// a byte count via /8), bytes 5..6 hold the slice count, byte 7 high
/// nibble is `log2_desired_slice_size_in_mb`.
fn parse_picture(frame: &[u8]) -> Picture {
    let hdr_size = u16::from_be_bytes([frame[8], frame[9]]) as usize;
    // Frame-header width/height (header-relative bytes 8/10 → frame bytes
    // 16/18 after the 8-byte container preamble).
    let width = u16::from_be_bytes([frame[16], frame[17]]) as usize;
    let height = u16::from_be_bytes([frame[18], frame[19]]) as usize;

    let pic = 8 + hdr_size;
    let pic_hdr_bits = frame[pic] as usize;
    assert_eq!(pic_hdr_bits % 8, 0, "picture header must be byte-aligned");
    let pic_hdr_len = pic_hdr_bits / 8;
    let num_slices = u16::from_be_bytes([frame[pic + 5], frame[pic + 6]]) as usize;
    let mbs_per_slice = 1usize << (frame[pic + 7] >> 4);

    let table = pic + pic_hdr_len;
    let mut slice_sizes = Vec::with_capacity(num_slices);
    for s in 0..num_slices {
        slice_sizes
            .push(u16::from_be_bytes([frame[table + s * 2], frame[table + s * 2 + 1]]) as usize);
    }
    let first_slice_off = table + num_slices * 2;

    Picture {
        mbs_per_slice,
        num_slices,
        slice_sizes,
        first_slice_off,
        width,
        height,
    }
}

/// Extract the raw `scanned_alpha()` blob and the MB count of slice
/// `idx`. The per-slice header byte 0's high 5 bits are the slice header
/// size; bytes 2..3 / 4..5 / 6..7 are Y / Cb / Cr coded sizes; alpha is
/// the remainder of the slice payload.
fn slice_alpha<'a>(frame: &'a [u8], pic: &Picture, idx: usize) -> &'a [u8] {
    let mut off = pic.first_slice_off;
    for &sz in &pic.slice_sizes[..idx] {
        off += sz;
    }
    let slice = &frame[off..off + pic.slice_sizes[idx]];
    let shs = (slice[0] >> 3) as usize;
    let y = u16::from_be_bytes([slice[2], slice[3]]) as usize;
    let u = u16::from_be_bytes([slice[4], slice[5]]) as usize;
    let v = u16::from_be_bytes([slice[6], slice[7]]) as usize;
    &slice[shs + y + u + v..]
}

/// Build the full-frame 12-bit alpha plane independently from the
/// bitstream, honouring §7.5.3 placement: each slice's array is the full
/// `16 * mbs_per_slice` columns × 16 rows; excess bottom rows (when the
/// picture height is not a multiple of MB_SIDE_PX) and excess right
/// columns (when the width is not MB-aligned) are discarded.
///
/// Returns a tight `width * height` plane of 12-bit samples.
fn reconstruct_alpha_plane(frame: &[u8], pic: &Picture) -> Vec<u16> {
    let width = pic.width;
    let height = pic.height;
    let mut plane = vec![0u16; width * height];

    let mbs_x = width.div_ceil(MB_SIDE_PX);
    let slices_per_row = mbs_x.div_ceil(pic.mbs_per_slice);
    assert_eq!(
        pic.num_slices % slices_per_row,
        0,
        "slice count must be a whole number of MB rows"
    );

    for idx in 0..pic.num_slices {
        let mb_row = idx / slices_per_row;
        let col_slice = idx % slices_per_row;
        // MBs covered by this slice (the last slice in a row may be
        // narrower if mbs_x is not a multiple of mbs_per_slice; this
        // fixture is 120 MBs / 8 = 15 even slices, so always 8).
        let mx = col_slice * pic.mbs_per_slice;
        let mbs_this = pic.mbs_per_slice.min(mbs_x - mx);
        let cols = mbs_this * MB_SIDE_PX;

        let blob = slice_alpha(frame, pic, idx);
        let values = decode_scanned_alpha(blob, cols * MB_SIDE_PX, AlphaChannelType::Sixteen)
            .unwrap_or_else(|e| panic!("slice {idx} alpha decode: {e}"));

        let y0 = mb_row * MB_SIDE_PX;
        let x0 = mx * MB_SIDE_PX;
        for r in 0..MB_SIDE_PX {
            let frame_row = y0 + r;
            if frame_row >= height {
                break; // §7.5.3: excess bottom rows discarded.
            }
            for c in 0..cols {
                let frame_col = x0 + c;
                if frame_col >= width {
                    break; // §7.5.3: excess right columns discarded.
                }
                plane[frame_row * width + frame_col] = promote_16_to_12(values[r * cols + c]);
            }
        }
    }
    plane
}

fn fixture_mov() -> Option<Vec<u8>> {
    let p = PathBuf::from("../../docs/video/prores/fixtures/4444-with-alpha/input.mov");
    match fs::read(&p) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!(
                "skip: missing {} ({e}). docs/ fixtures live in the workspace \
                 umbrella repo — the standalone crate checkout has no corpus.",
                p.display()
            );
            None
        }
    }
}

/// The decoder's emitted alpha plane (plane 3, 12-bit LE) must be
/// byte-for-byte identical to the independent §7.5.2/§7.5.3
/// reconstruction from the same bitstream.
#[test]
fn decoder_alpha_plane_matches_independent_reconstruction() {
    let Some(mov) = fixture_mov() else { return };
    let frame = first_prores_frame(&mov);
    let pic = parse_picture(frame);
    assert_eq!((pic.width, pic.height), (1920, 1080));
    assert_eq!(pic.mbs_per_slice, 8);
    assert_eq!(pic.num_slices, 1020);

    let reference = reconstruct_alpha_plane(frame, &pic);

    let vf = decode_packet_with_depth(frame, Some(0), Some((BitDepth::Twelve, ChromaFormat::Y444)))
        .expect("4444-with-alpha frame must decode");
    assert_eq!(vf.planes.len(), 4, "alpha-bearing frame must emit 4 planes");

    let a = &vf.planes[3];
    // Plane 3 is 12-bit LE-packed; read it into u16 samples honouring its
    // byte stride (which may exceed width*2 for the padded plane → here
    // the decoder crops to width, so stride == width*2, but read defensively).
    let bps = 2usize;
    let stride_samples = a.stride / bps;
    assert!(
        stride_samples >= pic.width,
        "alpha stride {} samples < width {}",
        stride_samples,
        pic.width
    );

    let mut mismatches = 0usize;
    let mut first: Option<(usize, usize, u16, u16)> = None;
    for y in 0..pic.height {
        let row = &a.data[y * a.stride..y * a.stride + pic.width * bps];
        for x in 0..pic.width {
            let got = u16::from_le_bytes([row[x * 2], row[x * 2 + 1]]);
            let want = reference[y * pic.width + x];
            if got != want {
                if first.is_none() {
                    first = Some((x, y, got, want));
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "decoder alpha plane diverged from the independent §7.5.2/§7.5.3 \
         reconstruction in {mismatches} samples; first at {:?} (got vs want)",
        first
    );
}

/// Cross-check the §7.5.2 promotion endpoints used above so the
/// reconstruction's `promote_16_to_12` is itself pinned to the spec rule
/// (and so a silent change to the rounding doesn't make the comparison
/// vacuously pass on a uniform plane).
#[test]
fn promotion_endpoints_match_spec() {
    // 16-bit fully transparent → 0; fully opaque 0xFFFF → 4095.
    assert_eq!(promote_16_to_12(0), 0);
    assert_eq!(promote_16_to_12(0xFFFF), 4095);
    // Midpoint 0x8000 → round(4095 * 32768 / 65535) = round(2047.53) = 2048.
    assert_eq!(promote_16_to_12(0x8000), 2048);
    // 0x4000 → round(4095 * 16384 / 65535) = round(1023.77) = 1024.
    assert_eq!(promote_16_to_12(0x4000), 1024);
}
