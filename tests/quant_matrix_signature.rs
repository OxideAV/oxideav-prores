//! Cross-validation of the per-profile signature quantisation matrices
//! (`quant::SIGNATURE_*_QMAT`, `QuantMatrices::signature_for_profile`)
//! against the reference ProRes streams in the in-tree corpus.
//!
//! Each `docs/video/prores/fixtures/<profile>/input.mov` carries one
//! ProRes frame whose frame header declares the profile's native
//! quantisation matrix pair. This suite decodes each fixture's frame
//! header with this crate's own parser (no external decoder consulted —
//! the fixtures are opaque data) and asserts:
//!
//! 1. the carried `luma_qmat` / `chroma_qmat` match the crate's
//!    `SIGNATURE_*_QMAT` constants byte-for-byte, and
//! 2. `QuantMatrices::signature_for_profile(profile)` reproduces exactly
//!    that pair, and
//! 3. re-encoding a frame header with the signature preset carries the
//!    same matrices back onto the wire (`from_header` round-trip) with
//!    the expected `(load_luma, load_chroma)` carriage.
//!
//! This locks the hard-coded signature constants to the reference bytes,
//! so a transcription slip in the tables cannot pass CI, and pins that
//! ProRes stores the quantisation matrix in natural (row-major) order —
//! the order the decoder applies it in (see `signature_matrices_are_
//! low_to_high_frequency_monotone` in the `quant` unit tests).

use std::fs;
use std::path::PathBuf;

use oxideav_prores::frame::{parse_frame, Profile};
use oxideav_prores::quant::{self, QuantMatrices};

/// Read a fixture frame, or `None` when the `docs/` corpus submodule is
/// not checked out (the standalone crate CI has no fixtures — a missing
/// file is a skip, not a failure, matching `docs_corpus.rs`).
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
    panic!("no ProRes frame found in {name}");
}

/// Decode a fixture's frame header and return the carried
/// `(luma_qmat, chroma_qmat)` (with the §6.1.1 chroma-copies-luma
/// fallback already applied by the parser), or `None` when the fixture
/// is absent.
fn carried_matrices(name: &str) -> Option<([u8; 64], [u8; 64], bool, bool)> {
    let frame = fixture_frame(name)?;
    let (fh, _) = parse_frame(&frame).expect("parse frame header");
    Some((
        fh.luma_qmat,
        fh.chroma_qmat,
        fh.load_luma_quantization_matrix,
        fh.load_chroma_quantization_matrix,
    ))
}

fn check(name: &str, profile: Profile, exp_luma: &[u8; 64], exp_chroma: &[u8; 64]) {
    let Some(frame) = fixture_frame(name) else {
        return;
    };
    let (fh, _) = parse_frame(&frame).expect("parse frame header");
    assert_eq!(&fh.luma_qmat, exp_luma, "{name}: carried luma matrix");
    assert_eq!(&fh.chroma_qmat, exp_chroma, "{name}: carried chroma matrix");

    // The signature preset must reproduce exactly the carried pair.
    let sig = QuantMatrices::signature_for_profile(profile);
    assert_eq!(&sig.luma, exp_luma, "{name}: signature luma");
    assert_eq!(&sig.chroma, exp_chroma, "{name}: signature chroma");

    // from_header on the parsed fixture recovers the same pair, and its
    // wire_flags reproduce the signature preset's carriage.
    let recovered = QuantMatrices::from_header(&fh);
    assert_eq!(recovered, sig, "{name}: from_header == signature preset");
}

#[test]
fn proxy_signature_matches_corpus() {
    check(
        "proxy-1280x720",
        Profile::Proxy,
        &quant::SIGNATURE_PROXY_LUMA_QMAT,
        &quant::SIGNATURE_PROXY_CHROMA_QMAT,
    );
    // Proxy is the only profile that carries a distinct chroma table.
    if let Some((_, _, load_l, load_c)) = carried_matrices("proxy-1280x720") {
        assert!(load_l && load_c, "proxy carries both quant tables");
    }
}

#[test]
fn lt_signature_matches_corpus() {
    check(
        "lt-1280x720",
        Profile::Lt,
        &quant::SIGNATURE_LT_QMAT,
        &quant::SIGNATURE_LT_QMAT,
    );
}

#[test]
fn standard_signature_matches_corpus() {
    check(
        "sq-1920x1080",
        Profile::Standard,
        &quant::SIGNATURE_STANDARD_QMAT,
        &quant::SIGNATURE_STANDARD_QMAT,
    );
}

#[test]
fn hq_signature_matches_corpus() {
    check(
        "hq-1920x1080",
        Profile::Hq,
        &quant::SIGNATURE_HQ_QMAT,
        &quant::SIGNATURE_HQ_QMAT,
    );
}

#[test]
fn prores4444_signature_matches_corpus() {
    check(
        "4444-1920x1080",
        Profile::Prores4444,
        &quant::SIGNATURE_HQ_QMAT,
        &quant::SIGNATURE_HQ_QMAT,
    );
}

#[test]
fn prores4444xq_signature_matches_corpus() {
    check(
        "4444xq-1920x1080",
        Profile::Prores4444Xq,
        &quant::SIGNATURE_HQ_QMAT,
        &quant::SIGNATURE_HQ_QMAT,
    );
}

/// The high-quality signature table is shared by HQ / 4444 / 4444 XQ —
/// all three corpus fixtures must carry the identical matrix.
#[test]
fn hq_family_carries_one_shared_matrix() {
    let (Some((hq, ..)), Some((v4444, ..)), Some((xq, ..))) = (
        carried_matrices("hq-1920x1080"),
        carried_matrices("4444-1920x1080"),
        carried_matrices("4444xq-1920x1080"),
    ) else {
        return;
    };
    assert_eq!(hq, v4444);
    assert_eq!(hq, xq);
    assert_eq!(&hq, &quant::SIGNATURE_HQ_QMAT);
}
