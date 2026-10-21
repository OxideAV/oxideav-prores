//! 4:4:4 chroma blocks within a macroblock are coded in the same order as
//! the luma blocks are not: the four 8x8 chroma blocks of a 16x16
//! macroblock come top-left, bottom-left, top-right, bottom-right.
//!
//! The fixture is one 16x16 4:4:4 10-bit frame from an independent
//! encoder whose Cb quadrants are 200 / 400 / 600 / 800 and Cr quadrants
//! 800 / 600 / 400 / 200 (top-left, top-right, bottom-left,
//! bottom-right), luma flat at 512.

use oxideav_prores::decoder::{decode_packet_with_options, BitDepth, OutputRange};
use oxideav_prores::frame::ChromaFormat;

const FIXTURE: &[u8] = include_bytes!("fixtures/chroma444_quadrants_16x16.prores");

fn sample(plane: &oxideav_core::frame::VideoPlane, x: usize, y: usize) -> u16 {
    let o = y * plane.stride + x * 2;
    u16::from_le_bytes([plane.data[o], plane.data[o + 1]])
}

#[test]
fn chroma_quadrants_land_where_they_were_coded() {
    let frame = decode_packet_with_options(FIXTURE, None, Some((BitDepth::Ten, ChromaFormat::Y444)), OutputRange::Full).expect("decode");
    let planes = frame.image_planes();
    let corners = [(0, 0), (8, 0), (0, 8), (8, 8)];
    for (p, want) in [(1, [200u16, 400, 600, 800]), (2, [800, 600, 400, 200])] {
        let got: Vec<u16> = corners.iter().map(|&(x, y)| sample(&planes[p], x + 3, y + 3)).collect();
        for (g, w) in got.iter().zip(want) {
            assert!(g.abs_diff(w) <= 2, "plane {p}: quadrants {got:?}, want {want:?}");
        }
    }
}
