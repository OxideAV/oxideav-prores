//! The opt-in entry points that keep reconstructed samples beyond the
//! visible picture: decode_packet_with_chroma_border (one more 4:2:2
//! chroma column when the visible width ends inside a macroblock) and
//! decode_packet_with_coded_width (the whole right macroblock).
//!
//! The fixture is one 20x16 10-bit 4:2:2 frame from an independent
//! encoder (two macroblocks wide, 12 columns of padding).

use oxideav_prores::decoder::{
    BitDepth, OutputRange, decode_packet_with_chroma_border, decode_packet_with_coded_width,
    decode_packet_with_options,
};
use oxideav_prores::frame::ChromaFormat;

const FIXTURE: &[u8] = include_bytes!("fixtures/edge_422_20x16.prores");
const TEN: Option<(BitDepth, ChromaFormat)> = Some((BitDepth::Ten, ChromaFormat::Y422));

fn row(p: &oxideav_core::frame::VideoPlane, y: usize, n: usize) -> Vec<u16> {
    (0..n)
        .map(|x| {
            u16::from_le_bytes([
                p.data[y * p.stride + 2 * x],
                p.data[y * p.stride + 2 * x + 1],
            ])
        })
        .collect()
}

#[test]
fn chroma_border_keeps_one_more_chroma_column() {
    let plain = decode_packet_with_options(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let wide = decode_packet_with_chroma_border(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let (p, w) = (plain.image_planes(), wide.image_planes());
    assert_eq!(p[0].stride, w[0].stride, "luma unchanged");
    assert_eq!(p[0].data, w[0].data);
    for c in 1..3 {
        assert_eq!(p[c].stride, 20, "10 visible chroma samples");
        assert_eq!(w[c].stride, 22, "one retained chroma sample");
        for y in 0..16 {
            assert_eq!(row(&p[c], y, 10), row(&w[c], y, 10)[..10]);
        }
    }
}

#[test]
fn coded_width_keeps_the_right_macroblock() {
    let plain = decode_packet_with_options(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let coded = decode_packet_with_coded_width(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let (p, c) = (plain.image_planes(), coded.image_planes());
    assert_eq!(c[0].stride, 64, "32 coded luma samples");
    assert_eq!(c[1].stride, 32, "16 coded chroma samples");
    for y in 0..16 {
        assert_eq!(row(&p[0], y, 20), row(&c[0], y, 20)[..20]);
    }
}

/// The retained chroma column is the decoder's own reconstruction of
/// the next coded sample: it equals column 10 of the coded-width decode,
/// and the visible samples of all three decodes agree. (This fixture's
/// encoder pads by replicating the edge, so the value may equal column
/// 9; what matters is where it comes from.)
#[test]
fn retained_chroma_column_is_the_coded_sample() {
    let plain = decode_packet_with_options(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let wide = decode_packet_with_chroma_border(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let coded = decode_packet_with_coded_width(FIXTURE, None, TEN, OutputRange::Full).unwrap();
    let (p, w, c) = (
        plain.image_planes(),
        wide.image_planes(),
        coded.image_planes(),
    );
    for ch in 1..3 {
        for y in 0..16 {
            let wide_row = row(&w[ch], y, 11);
            let coded_row = row(&c[ch], y, 16);
            assert_eq!(wide_row[..10], row(&p[ch], y, 10), "visible samples");
            assert_eq!(coded_row[..10], row(&p[ch], y, 10), "visible samples");
            assert_eq!(
                wide_row[10], coded_row[10],
                "retained sample is the coded one"
            );
        }
    }
}
