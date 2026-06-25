//! RDD 36 §6.4 forward-compatibility: a decoder must locate the syntax
//! structure following a `picture()` from the **declared** `picture_size`
//! (§6.2.1), not by inferring the picture's end from the slice table it
//! parsed. §6.4 permits a future *version variant* to append informative
//! bytes after a picture's defined syntax; those bytes inflate
//! `picture_size` beyond `picture_header_size + slice_table + Σslice`.
//!
//! These tests synthesise exactly that situation by injecting filler
//! bytes after a picture's slices and bumping the relevant size fields,
//! then confirming the decoder:
//!   * still decodes the modified stream, and
//!   * produces byte-identical output to the unmodified stream, and
//!   * (interlaced) finds the second field at the correct offset even
//!     when the first field carries trailing variant bytes.

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::decoder::{decode_packet, BitDepth};
use oxideav_prores::encoder::{encode_frame_422, encode_frame_interlaced};
use oxideav_prores::frame::{ChromaFormat, Profile};

/// Build a small smooth 4:2:2 8-bit source frame.
fn source_422(width: usize, height: usize) -> VideoFrame {
    let cw = width.div_ceil(2);
    let mut y = vec![0u8; width * height];
    let mut cb = vec![0u8; cw * height];
    let mut cr = vec![0u8; cw * height];
    for j in 0..height {
        for i in 0..width {
            y[j * width + i] = ((i * 5 + j * 3) % 256) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (96 + ((i + j) % 64)) as u8;
            cr[j * cw + i] = (160 - ((i + j) % 64)) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: width,
                data: y,
            },
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

/// Read the big-endian u32 at `off`.
fn be32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(buf[off..off + 4].try_into().unwrap())
}

/// Write the big-endian u32 `v` at `off`.
fn put_be32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
}

/// Offset of the first `picture()` within a ProRes frame: 8-byte
/// container preamble (`frame_size` + `'icpf'`) + the frame header. For
/// these flat-quant streams the frame header is the minimal 20 bytes.
fn first_picture_offset(stream: &[u8]) -> usize {
    let frame_header_size = u16::from_be_bytes(stream[8..10].try_into().unwrap()) as usize;
    8 + frame_header_size
}

/// Inject `extra` filler bytes immediately after the `picture()` that
/// begins at `pic_off`, simulating §6.4 version-variant trailing data.
/// Patches that picture's `picture_size` field (picture-header bytes
/// 1..5) and the outer `frame_size` (stream bytes 0..4) so the stream
/// stays internally consistent. Returns the rewritten stream.
fn inject_variant_bytes(stream: &[u8], pic_off: usize, extra: usize) -> Vec<u8> {
    // picture_size lives at picture-header offset 1 (u32, big-endian).
    let old_pic_size = be32(stream, pic_off + 1) as usize;
    let insert_at = pic_off + old_pic_size;

    let mut out = Vec::with_capacity(stream.len() + extra);
    out.extend_from_slice(&stream[..insert_at]);
    // A non-zero, recognisable filler that the decoder must skip without
    // interpreting.
    out.extend(std::iter::repeat(0xA5u8).take(extra));
    out.extend_from_slice(&stream[insert_at..]);

    // Bump the picture's declared size and the frame's declared size.
    put_be32(&mut out, pic_off + 1, (old_pic_size + extra) as u32);
    let old_frame_size = be32(&out, 0) as usize;
    put_be32(&mut out, 0, (old_frame_size + extra) as u32);
    out
}

fn planes_eq(a: &VideoFrame, b: &VideoFrame) -> bool {
    a.planes.len() == b.planes.len()
        && a.planes
            .iter()
            .zip(&b.planes)
            .all(|(p, q)| p.stride == q.stride && p.data == q.data)
}

#[test]
fn progressive_picture_trailing_variant_bytes_skipped() {
    let src = source_422(80, 48);
    let stream =
        encode_frame_422(&src, 80, 48, Profile::Standard, 4).expect("encode progressive 422");
    let baseline = decode_packet(&stream, Some(0)).expect("decode unmodified");

    let pic_off = first_picture_offset(&stream);
    for extra in [1usize, 4, 31, 32, 256] {
        let modified = inject_variant_bytes(&stream, pic_off, extra);
        let got = decode_packet(&modified, Some(0))
            .unwrap_or_else(|e| panic!("decode with {extra} variant bytes failed: {e:?}"));
        assert!(
            planes_eq(&got, &baseline),
            "output differed after injecting {extra} trailing variant bytes"
        );
    }
}

#[test]
fn interlaced_first_field_trailing_variant_bytes_skipped() {
    // 80x48 TFF: two pictures (top field then bottom field). Injecting
    // variant bytes after the FIRST field exercises the §6.2.1 rule that
    // the second field's start is taken from the first picture's declared
    // `picture_size`.
    let src = source_422(80, 48);
    let stream = encode_frame_interlaced(
        &src,
        80,
        48,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        None,
        1,
    )
    .expect("encode interlaced 422 TFF");
    let baseline = decode_packet(&stream, Some(0)).expect("decode unmodified interlaced");

    let first_pic_off = first_picture_offset(&stream);
    for extra in [1usize, 8, 64] {
        let modified = inject_variant_bytes(&stream, first_pic_off, extra);
        let got = decode_packet(&modified, Some(0)).unwrap_or_else(|e| {
            panic!("interlaced decode with {extra} first-field variant bytes failed: {e:?}")
        });
        assert!(
            planes_eq(&got, &baseline),
            "interlaced output differed after {extra} first-field variant bytes"
        );
    }
}

/// Read the big-endian u16 at `off`.
fn be16(buf: &[u8], off: usize) -> u16 {
    u16::from_be_bytes(buf[off..off + 2].try_into().unwrap())
}

/// Write the big-endian u16 `v` at `off`.
fn put_be16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_be_bytes());
}

/// Inject `extra` filler bytes immediately after the **frame header's**
/// defined syntax (i.e. just before the first `picture()`), simulating a
/// §6.1.1 / §6.4 version variant that appends informative bytes inside the
/// frame header. RDD 36 §6.1.1 (`frame_header_size`): "Decoders shall use
/// this value to determine the start of the compressed picture following
/// the frame header in the bitstream." A conforming decoder must therefore
/// locate the first picture from the *declared* `frame_header_size`, not by
/// inferring the header's end from the fields it parsed.
///
/// Patches `frame_header_size` (stream bytes 0..2 of the frame header, i.e.
/// stream bytes 8..10) and the outer `frame_size` (stream bytes 0..4).
fn inject_frame_header_variant_bytes(stream: &[u8], extra: usize) -> Vec<u8> {
    // frame() preamble is 8 bytes (frame_size u32 + 'icpf'); the frame
    // header begins at stream offset 8 and its first field is
    // frame_header_size (u16, big-endian).
    let fh_off = 8usize;
    let old_fh_size = be16(stream, fh_off) as usize;
    let insert_at = fh_off + old_fh_size;

    let mut out = Vec::with_capacity(stream.len() + extra);
    out.extend_from_slice(&stream[..insert_at]);
    out.extend(std::iter::repeat(0x5Au8).take(extra));
    out.extend_from_slice(&stream[insert_at..]);

    put_be16(&mut out, fh_off, (old_fh_size + extra) as u16);
    let old_frame_size = be32(&out, 0) as usize;
    put_be32(&mut out, 0, (old_frame_size + extra) as u32);
    out
}

#[test]
fn frame_header_trailing_variant_bytes_skipped() {
    // A §6.1.1 version variant appends informative bytes after the frame
    // header's defined syntax. The decoder must find the first picture from
    // the declared `frame_header_size` and decode byte-identically.
    let src = source_422(80, 48);
    let stream =
        encode_frame_422(&src, 80, 48, Profile::Standard, 4).expect("encode progressive 422");
    let baseline = decode_packet(&stream, Some(0)).expect("decode unmodified");

    for extra in [1usize, 4, 11, 64, 256] {
        let modified = inject_frame_header_variant_bytes(&stream, extra);
        let got = decode_packet(&modified, Some(0)).unwrap_or_else(|e| {
            panic!("decode with {extra} frame-header variant bytes failed: {e:?}")
        });
        assert!(
            planes_eq(&got, &baseline),
            "output differed after injecting {extra} frame-header variant bytes"
        );
    }
}

#[test]
fn frame_header_variant_bytes_skipped_interlaced() {
    // The same §6.1.1 frame-header version variant on an interlaced frame:
    // the *single* frame header precedes both fields, so finding it short
    // would mis-locate the very first picture and cascade into both fields.
    let src = source_422(80, 48);
    let stream = encode_frame_interlaced(
        &src,
        80,
        48,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        None,
        1,
    )
    .expect("encode interlaced 422 TFF");
    let baseline = decode_packet(&stream, Some(0)).expect("decode unmodified interlaced");

    for extra in [1usize, 7, 64] {
        let modified = inject_frame_header_variant_bytes(&stream, extra);
        let got = decode_packet(&modified, Some(0)).unwrap_or_else(|e| {
            panic!("interlaced decode with {extra} frame-header variant bytes failed: {e:?}")
        });
        assert!(
            planes_eq(&got, &baseline),
            "interlaced output differed after {extra} frame-header variant bytes"
        );
    }
}

/// Inject `extra` filler bytes immediately after the **first slice
/// header's** defined syntax (i.e. just before that slice's compressed
/// luma data), simulating a §6.3.1 / §6.4 version variant that appends
/// informative bytes inside a slice header. RDD 36 §6.3.1
/// (`slice_header_size`): "Decoders shall use this value to determine the
/// start of the compressed luma component data following the slice header
/// in the bitstream." A conforming decoder must therefore locate the Y
/// data from the *declared* `slice_header_size`, not from the sum of the
/// fixed slice-header fields it parsed.
///
/// Bumps, in lock-step, every length that frames this slice: the first
/// slice's `slice_header_size` (high 5 bits of the slice's first byte), the
/// matching `coded_size_of_slice` table entry, the enclosing
/// `picture_size`, and the outer `frame_size`.
fn inject_slice_header_variant_bytes(stream: &[u8], extra: usize) -> Vec<u8> {
    let pic_off = first_picture_offset(stream);
    // Parse the picture header to learn its size and slice geometry.
    let (ph, _rest) = oxideav_prores::frame::parse_picture_header(&stream[pic_off..])
        .expect("parse picture header for slice-header injection");
    let phs = ph.picture_header_size as usize;

    // mbs_x = ceil(width / 16); width lives in the frame header at stream
    // bytes 16..18 (frame header offset 8 = horizontal_size).
    let width = be16(stream, 16) as usize;
    let mbs_x = width.div_ceil(16);
    let slices_per_row =
        oxideav_prores::frame::compute_slice_sizes(mbs_x, ph.log2_desired_slice_size_in_mb).len();
    // height at frame-header offset 10 -> stream bytes 18..20.
    let height = be16(stream, 18) as usize;
    let mbs_y = height.div_ceil(16);
    let total_slices = slices_per_row * mbs_y;
    let slice_table_bytes = total_slices * 2;

    let slice_table_off = pic_off + phs;
    let first_slice_off = slice_table_off + slice_table_bytes;
    let first_slice_size = be16(stream, slice_table_off) as usize;

    // slice_header_size is the high 5 bits of the slice's first byte.
    let shs = (stream[first_slice_off] >> 3) as usize & 0x1F;
    let insert_at = first_slice_off + shs;

    let mut out = Vec::with_capacity(stream.len() + extra);
    out.extend_from_slice(&stream[..insert_at]);
    out.extend(std::iter::repeat(0x3Cu8).take(extra));
    out.extend_from_slice(&stream[insert_at..]);

    // Bump slice_header_size (high 5 bits of the slice's first byte).
    let new_shs = shs + extra;
    assert!(new_shs <= 31, "slice_header_size must stay in 5 bits");
    let lo3 = out[first_slice_off] & 0x07;
    out[first_slice_off] = ((new_shs as u8) << 3) | lo3;
    // Bump this slice's coded_size_of_slice table entry.
    put_be16(&mut out, slice_table_off, (first_slice_size + extra) as u16);
    // Bump picture_size and frame_size.
    let old_pic_size = be32(&out, pic_off + 1) as usize;
    put_be32(&mut out, pic_off + 1, (old_pic_size + extra) as u32);
    let old_frame_size = be32(&out, 0) as usize;
    put_be32(&mut out, 0, (old_frame_size + extra) as u32);
    out
}

#[test]
fn slice_header_trailing_variant_bytes_skipped() {
    // A §6.3.1 version variant appends informative bytes after a slice
    // header's defined syntax. The decoder must find the slice's luma data
    // from the declared `slice_header_size` and decode byte-identically.
    // 4:2:2 with no alpha keeps the slice header at its 6-byte minimum so
    // the injected bytes are unambiguously variant data, not coded_size_*
    // fields.
    let src = source_422(80, 48);
    let stream =
        encode_frame_422(&src, 80, 48, Profile::Standard, 4).expect("encode progressive 422");
    let baseline = decode_packet(&stream, Some(0)).expect("decode unmodified");

    for extra in [1usize, 2, 9, 25] {
        let modified = inject_slice_header_variant_bytes(&stream, extra);
        let got = decode_packet(&modified, Some(0)).unwrap_or_else(|e| {
            panic!("decode with {extra} slice-header variant bytes failed: {e:?}")
        });
        assert!(
            planes_eq(&got, &baseline),
            "output differed after injecting {extra} slice-header variant bytes"
        );
    }
}

#[test]
fn picture_payload_exceeding_declared_size_is_rejected() {
    // The inverse guard: if the consumed syntax (header + slice table +
    // slices) would exceed the declared `picture_size`, the slice table
    // is corrupt and the decoder must refuse rather than read past the
    // picture. Shrink `picture_size` below the real payload to trigger it.
    let src = source_422(80, 48);
    let stream =
        encode_frame_422(&src, 80, 48, Profile::Standard, 4).expect("encode progressive 422");
    let pic_off = first_picture_offset(&stream);

    let mut bad = stream.clone();
    let real = be32(&bad, pic_off + 1);
    assert!(real > 16, "sanity: picture_size should be well above 16");
    put_be32(&mut bad, pic_off + 1, real - 16);
    let err = decode_packet(&bad, Some(0));
    assert!(
        err.is_err(),
        "decoder accepted a picture whose payload exceeds its declared picture_size"
    );
}
