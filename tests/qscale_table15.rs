//! RDD 36 §7.3 / Table 15 — quantization scale factor `qScale` as a
//! function of `quantization_index` — pinned exhaustively against the
//! spec's printed table values and its compact closed form.
//!
//! Table 15 is a two-segment piecewise map:
//!
//! ```text
//!   qScale = quantization_index                         for  1 ≤ qi ≤ 128
//!   qScale = 128 + 4 * (quantization_index − 128)       for 129 ≤ qi ≤ 224
//! ```
//!
//! The spec prints the anchor rows `1->1`, `126->126`, `127->127`,
//! `128->128`, `129->132`, `130->136`, `223->508`, `224->512`. The key
//! property is the discontinuity at the 128/129 boundary: the slope changes
//! from 1 to 4, so qScale jumps from 128 to 132 (a step of 4, not 1) as
//! `quantization_index` crosses from 128 to 129. A decoder that applies the
//! linear `qi` form past 128, or the `128 + 4*(qi-128)` form at or below
//! 128, would mis-dequantise every slice carrying a coarse quantization
//! index.
//!
//! This test is DATA-only: every expected value comes from RDD 36
//! Table 15 (a numeric spec table), not from any external codec.

use oxideav_prores::frame::SliceHeader;
use oxideav_prores::quant::qscale;

/// The eight rows RDD 36 prints explicitly in Table 15.
const TABLE_15_ANCHORS: &[(u8, i32)] = &[
    (1, 1),
    (126, 126),
    (127, 127),
    (128, 128),
    (129, 132),
    (130, 136),
    (223, 508),
    (224, 512),
];

#[test]
fn qscale_matches_table_15_printed_anchors() {
    for &(qi, expected) in TABLE_15_ANCHORS {
        assert_eq!(
            qscale(qi),
            expected,
            "Table 15 anchor row: quantization_index {qi} → qScale {expected}"
        );
    }
}

#[test]
fn qscale_lower_segment_is_identity() {
    // Segment 1: qScale == quantization_index for 1..=128.
    for qi in 1u8..=128 {
        assert_eq!(
            qscale(qi),
            qi as i32,
            "lower segment: quantization_index {qi} must map to itself"
        );
    }
}

#[test]
fn qscale_upper_segment_is_slope_four() {
    // Segment 2: qScale == 128 + 4 * (qi - 128) for 129..=224.
    for qi in 129u8..=224 {
        let expected = 128 + 4 * (qi as i32 - 128);
        assert_eq!(
            qscale(qi),
            expected,
            "upper segment: quantization_index {qi} must map to {expected}"
        );
    }
}

#[test]
fn qscale_is_discontinuous_at_the_128_129_boundary() {
    // The defining property of Table 15: a slope change from 1 to 4 at
    // the boundary. 128 → 128 (last of segment 1), 129 → 132 (first of
    // segment 2), so the step across the boundary is 4, not 1.
    assert_eq!(qscale(128), 128);
    assert_eq!(qscale(129), 132);
    assert_eq!(
        qscale(129) - qscale(128),
        4,
        "the 128→129 step must be 4 (slope-4 segment begins), not 1"
    );
    // Within segment 1 the step is 1; within segment 2 the step is 4.
    assert_eq!(qscale(128) - qscale(127), 1, "segment-1 step is 1");
    assert_eq!(qscale(130) - qscale(129), 4, "segment-2 step is 4");
}

#[test]
fn qscale_is_monotonically_increasing_over_the_defined_range() {
    // qScale strictly increases across 1..=224 (1..=512), so a coarser
    // quantization_index always selects a coarser scale.
    let mut prev = qscale(1);
    for qi in 2u8..=224 {
        let cur = qscale(qi);
        assert!(
            cur > prev,
            "qScale must strictly increase: qi {qi} gave {cur}, previous {prev}"
        );
        prev = cur;
    }
    // Endpoints: finest index 1 → qScale 1, coarsest 224 → qScale 512.
    assert_eq!(qscale(1), 1);
    assert_eq!(qscale(224), 512);
}

#[test]
fn slice_header_qscale_accessor_mirrors_table_15() {
    // The typed `SliceHeader::qscale()` accessor folds the same Table 15
    // derivation; a parsed-or-hand-built header in the defined index
    // range must return `Some(qScale)` equal to the free function.
    for qi in 1u8..=224 {
        let sh = SliceHeader {
            slice_header_size: 6,
            quantization_index: qi,
            coded_size_of_y_data: 0,
            coded_size_of_cb_data: 0,
            coded_size_of_cr_data: None,
        };
        assert_eq!(
            sh.qscale(),
            Some(qscale(qi)),
            "SliceHeader::qscale() must mirror quant::qscale for qi {qi}"
        );
    }
}

#[test]
fn slice_header_qscale_is_none_for_reserved_indices() {
    // §6.3.1 restricts quantization_index to 1..=224; 0 and 225..=255 are
    // reserved. A hand-built header carrying a reserved code returns
    // `None` (no defined qScale), distinct from `Some(1)`.
    for qi in std::iter::once(0u8).chain(225u8..=255) {
        let sh = SliceHeader {
            slice_header_size: 6,
            quantization_index: qi,
            coded_size_of_y_data: 0,
            coded_size_of_cb_data: 0,
            coded_size_of_cr_data: None,
        };
        assert_eq!(
            sh.qscale(),
            None,
            "reserved quantization_index {qi} must yield None"
        );
    }
}
