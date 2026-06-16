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
