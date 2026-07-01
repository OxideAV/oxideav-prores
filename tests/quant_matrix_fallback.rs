//! RDD 36 §6.1.1 quantization-matrix fallback — pixel-exact equivalence
//! between the compact carriage forms and their explicit both-tables
//! twins.
//!
//! §6.1.1 lets a frame omit a quantization table and have the decoder
//! substitute one:
//!
//! - `load_chroma_quantization_matrix == 0` ⇒ chroma reuses the luma
//!   matrix (itself the custom luma if `load_luma == 1`, else the §7.2
//!   all-4s default).
//! - `load_luma_quantization_matrix == 0` ⇒ luma is the §7.2 default.
//!
//! The corpus never carries the compact forms (every fixture is flags
//! `(1, 1)`), so this suite constructs each compact stream with our
//! encoder, then byte-splices in the omitted table(s) to build the
//! equivalent explicit `(1, 1)` stream — flipping the load bit and bumping
//! `frame_size` + `frame_header_size` — and asserts the decoder produces
//! **byte-identical** planes for both. The picture() payload is unchanged
//! and its offsets are self-relative, so the splice is well-formed. This
//! proves each fallback arm dequantises exactly as the explicit table
//! would. Validator-independent.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Frame, MediaType, PixelFormat, VideoFrame,
};
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::encode_frame_with_qmats;
use oxideav_prores::frame::{ChromaFormat, Profile};
use oxideav_prores::quant::{QuantMatrices, DEFAULT_QMAT, PERCEPTUAL_LUMA_QMAT};
use oxideav_prores::CODEC_ID_STR;

const W: u32 = 80;
const H: u32 = 48;
// Absolute byte offsets within a ProRes frame unit written by this
// crate's encoder: frame_size u32 at [0..4], 'icpf' at [4..8],
// frame_header_size u16 at [8..10]; the frame header body starts at
// offset 8, so the load-flags byte (header-relative offset 19) is at 27
// and the first quantization table (header-relative 20) begins at 28.
const FRAME_SIZE_OFF: usize = 0;
const FH_SIZE_OFF: usize = 8;
const FLAGS_OFF: usize = 27;
const FIRST_TABLE_OFF: usize = 28;

fn synth_422() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (((i * 7 + j * 3) as u8) ^ ((i * j) as u8)).wrapping_add(16);
        }
        for i in 0..cw {
            cb[j * cw + i] = 100 + ((i + j) as u8 & 0x3F);
            cr[j * cw + i] = (140u8).wrapping_sub((i as u8).wrapping_add(j as u8) & 0x3F);
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

fn encode(qm: QuantMatrices) -> Vec<u8> {
    encode_frame_with_qmats(
        &synth_422(),
        W,
        H,
        ChromaFormat::Y422,
        BitDepth::Eight,
        Profile::Standard,
        4,
        qm,
    )
    .expect("encode")
}

/// Splice `insert` (a 64-byte quantization table, or two) into the frame
/// header at absolute offset `at`, set the load-flags byte to `new_flags`,
/// and bump `frame_size` + `frame_header_size` by the inserted length.
fn splice_tables(pkt: &[u8], at: usize, insert: &[u8], new_flags: u8) -> Vec<u8> {
    let mut v = pkt.to_vec();
    v[FLAGS_OFF] = new_flags;
    v.splice(at..at, insert.iter().copied());
    let n = insert.len();
    let fh = u16::from_be_bytes([v[FH_SIZE_OFF], v[FH_SIZE_OFF + 1]]) + n as u16;
    v[FH_SIZE_OFF..FH_SIZE_OFF + 2].copy_from_slice(&fh.to_be_bytes());
    let fs = u32::from_be_bytes([v[0], v[1], v[2], v[3]]) + n as u32;
    v[FRAME_SIZE_OFF..FRAME_SIZE_OFF + 4].copy_from_slice(&fs.to_be_bytes());
    v
}

fn decode_planes(pkt: &[u8]) -> Vec<VideoPlane> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    let mut reg = CodecRegistry::new();
    oxideav_prores::register_codecs(&mut reg);
    let mut dec = reg.first_decoder(&params).expect("make_decoder");
    let mut p = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 30), pkt.to_vec());
    p.flags.keyframe = true;
    dec.send_packet(&p).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v.planes,
        _ => panic!("expected video frame"),
    }
}

fn assert_same_planes(a: &[VideoPlane], b: &[VideoPlane]) {
    assert_eq!(a.len(), b.len(), "plane count");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.stride, y.stride, "plane {i} stride");
        assert_eq!(x.data, y.data, "plane {i} data");
    }
}

/// `load_chroma == 0` (chroma copies the custom luma) decodes identically
/// to an explicit `(1, 1)` stream that carries the same table twice.
#[test]
fn chroma_copies_luma_equals_explicit() {
    // Compact: (custom luma, chroma == luma) -> flags (1, 0), one table.
    let compact = encode(QuantMatrices {
        luma: PERCEPTUAL_LUMA_QMAT,
        chroma: PERCEPTUAL_LUMA_QMAT,
    });
    // The carried luma table is [FIRST_TABLE_OFF .. +64]; duplicate it as an
    // explicit chroma table right after it, and flip load_chroma on.
    let luma_table: Vec<u8> = compact[FIRST_TABLE_OFF..FIRST_TABLE_OFF + 64].to_vec();
    let explicit = splice_tables(&compact, FIRST_TABLE_OFF + 64, &luma_table, 0b11);
    assert_same_planes(&decode_planes(&compact), &decode_planes(&explicit));
}

/// Both-default (`0, 0`) decodes identically to an explicit `(1, 1)`
/// stream carrying two all-4s tables — i.e. the §7.2 default really is
/// all-4s for both components.
#[test]
fn both_default_equals_explicit_all_fours() {
    let compact = encode(QuantMatrices::flat()); // flags (0, 0), no tables.
    let mut two_tables = Vec::with_capacity(128);
    two_tables.extend_from_slice(&DEFAULT_QMAT);
    two_tables.extend_from_slice(&DEFAULT_QMAT);
    let explicit = splice_tables(&compact, FIRST_TABLE_OFF, &two_tables, 0b11);
    assert_same_planes(&decode_planes(&compact), &decode_planes(&explicit));
}

/// Default-luma + custom-chroma (`0, 1`) decodes identically to an
/// explicit `(1, 1)` stream that carries an all-4s luma table before the
/// custom chroma table — i.e. the omitted luma really falls back to the
/// §7.2 default.
#[test]
fn default_luma_fallback_equals_explicit() {
    // Compact: (default luma, custom chroma) -> flags (0, 1), chroma only.
    let mut chroma = DEFAULT_QMAT;
    for (k, w) in chroma.iter_mut().enumerate() {
        *w = (2 + (k % 60)) as u8; // a distinct valid 2..=61 chroma matrix
    }
    let compact = encode(QuantMatrices {
        luma: DEFAULT_QMAT,
        chroma,
    });
    // Insert an explicit all-4s luma table BEFORE the carried chroma table
    // and flip load_luma on.
    let explicit = splice_tables(&compact, FIRST_TABLE_OFF, &DEFAULT_QMAT, 0b11);
    assert_same_planes(&decode_planes(&compact), &decode_planes(&explicit));
}
