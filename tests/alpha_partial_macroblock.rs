//! Compact alpha on a partial bottom macroblock row.
//!
//! The fixture is one 32x20 ProRes 4444 frame with alpha from an
//! independent encoder (macOS's). Its last macroblock row has 4 visible
//! lines, and that encoder codes alpha for the visible lines only, so
//! that row's alpha payload holds 4 lines of values instead of 16.

use oxideav_prores::decoder::{OutputRange, decode_packet_with_options};

const FIXTURE: &[u8] = include_bytes!("fixtures/alpha_partial_mb_32x20.prores");

#[test]
fn partial_bottom_macroblock_alpha_decodes() {
    let frame = decode_packet_with_options(FIXTURE, None, None, OutputRange::Full).expect("decode");
    let planes = frame.image_planes();
    assert_eq!(planes.len(), 4, "Y, Cb, Cr, alpha");
    // The source alpha is (8x + 5y) mod 256 (8-bit), so it rises along
    // each row up to x = 20; check the partial rows carry it.
    let a = &planes[3];
    let bytes = a.stride / 32;
    let at = |x: usize, y: usize| -> u32 {
        let o = y * a.stride + bytes * x;
        if bytes == 1 {
            u32::from(a.data[o])
        } else {
            u32::from(u16::from_le_bytes([a.data[o], a.data[o + 1]]))
        }
    };
    for y in 16..20 {
        assert!(
            at(20, y) > at(2, y),
            "row {y}: alpha {} then {}",
            at(2, y),
            at(20, y)
        );
    }
}
