//! Black-box guard for the 4:4:4 chroma block order (RDD 36 §7.5,
//! Figure 8: within a macroblock the four Cb / Cr blocks are indexed
//! top-left, bottom-left, top-right, bottom-right — a column order that
//! differs from the luma order of Figure 6).
//!
//! The crate's own 4444 round trip cannot see a wrong order because the
//! encoder and the decoder would agree with each other, so this test
//! crosses with an independent implementation (`ffmpeg`, as a black-box
//! binary) in both directions and scores the CHROMA planes:
//!
//! 1. `prores_ks` 4444 encode of a detailed source → our decoder, against
//!    ffmpeg's own decode of the same packet;
//! 2. our 4444 encode of that source → ffmpeg decode, against the source.
//!
//! A swapped top-right / bottom-left pair lands the chroma PSNR in the
//! 25–35 dB range on detailed content; a matching order is limited only
//! by quantisation and IDCT precision (≥ 55 dB on this material).
//!
//! Skips when `ffmpeg` is missing.

use std::process::Command;

use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_prores::decoder::{BitDepth, OutputRange, decode_packet_with_options};
use oxideav_prores::encoder::encode_frame_with_depth;
use oxideav_prores::frame::{ChromaFormat, Profile};

const W: u32 = 64;
const H: u32 = 48;
const MIN_CHROMA_PSNR_DB: f64 = 55.0;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tempdir() -> std::path::PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let p = std::env::temp_dir().join(format!(
        "oxideav-prores-444-order-{}-{ts}",
        std::process::id()
    ));
    std::fs::create_dir_all(&p).expect("tempdir");
    p
}

fn ffmpeg_ok(args: &[&str]) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// The `icpf`-prefixed frame inside a 1-sample MOV's `mdat`.
fn extract_prores_packet(mov: &[u8]) -> Vec<u8> {
    let i = mov
        .windows(4)
        .position(|w| w == b"icpf")
        .expect("icpf magic in MOV");
    let size_off = i - 4;
    let frame_size = u32::from_be_bytes(mov[size_off..size_off + 4].try_into().unwrap()) as usize;
    mov[size_off..size_off + frame_size].to_vec()
}

fn find_top_atom(buf: &[u8], name: &[u8; 4]) -> Option<(usize, u32)> {
    let mut i = 0;
    while i + 8 <= buf.len() {
        let size = u32::from_be_bytes(buf[i..i + 4].try_into().unwrap());
        if &buf[i + 4..i + 8] == name {
            return Some((i, size));
        }
        if size < 8 {
            return None;
        }
        i += size as usize;
    }
    None
}

/// Substitute our packet into a 1-sample template MOV: rewrites the
/// `mdat` size and the single `stsz` entry (the `stco` offset stays
/// valid because the mdat payload keeps its file offset).
fn patch_mov_with_packet(template: &[u8], pkt: &[u8]) -> Vec<u8> {
    let (mdat_off, mdat_size) = find_top_atom(template, b"mdat").expect("mdat");
    let payload_start = mdat_off + 8;
    let payload_end = mdat_off + mdat_size as usize;
    let mut out = Vec::with_capacity(template.len() + pkt.len());
    out.extend_from_slice(&template[..payload_start]);
    out.extend_from_slice(pkt);
    out.extend_from_slice(&template[payload_end..]);
    out[mdat_off..mdat_off + 4].copy_from_slice(&((8 + pkt.len()) as u32).to_be_bytes());

    let stsz_off = out.windows(4).position(|w| w == b"stsz").expect("stsz");
    let sample_size = u32::from_be_bytes(out[stsz_off + 8..stsz_off + 12].try_into().unwrap());
    let entry_off = if sample_size != 0 {
        stsz_off + 8
    } else {
        stsz_off + 16
    };
    out[entry_off..entry_off + 4].copy_from_slice(&(pkt.len() as u32).to_be_bytes());
    out
}

/// PSNR between two LE-packed `u16` planes; `b_shift` left-shifts `b`'s
/// samples first (a 10-bit source against a 12-bit decode).
fn psnr_le16(a: &[u8], b: &[u8], b_shift: u32, peak: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "plane length");
    let n = a.len() / 2;
    let mut sse = 0.0f64;
    for i in 0..n {
        let av = u16::from_le_bytes([a[i * 2], a[i * 2 + 1]]) as f64;
        let bv = (u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) << b_shift) as f64;
        sse += (av - bv) * (av - bv);
    }
    let mse = sse / n as f64;
    if mse == 0.0 {
        return 120.0;
    }
    10.0 * (peak * peak / mse).log10()
}

/// Raw `yuv444p10le` frame → three planes of `W*H*2` bytes each.
fn planes_of(raw: &[u8]) -> [&[u8]; 3] {
    let pb = (W * H * 2) as usize;
    assert!(raw.len() >= pb * 3, "raw yuv444p10le shorter than 3 planes");
    [&raw[..pb], &raw[pb..pb * 2], &raw[pb * 2..pb * 3]]
}

#[test]
fn chroma_444_block_order_crosses_with_ffmpeg_both_ways() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping 4:4:4 chroma block order cross-check");
        return;
    }
    let tmp = tempdir();
    let src_path = tmp.join("src.yuv");
    let ff_mov = tmp.join("ff.mov");
    let ff_dec = tmp.join("ff_dec.yuv");
    let our_mov = tmp.join("ours.mov");
    let our_dec = tmp.join("ours_dec.yuv");
    let lavfi = format!("testsrc2=size={W}x{H}:rate=25:duration=1");

    // Source: one testsrc2 frame as raw yuv444p10le (detailed chroma).
    assert!(ffmpeg_ok(&[
        "-f",
        "lavfi",
        "-i",
        &lavfi,
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuv444p10le",
        "-f",
        "rawvideo",
        src_path.to_str().unwrap(),
    ]));
    let src = std::fs::read(&src_path).expect("read source");
    let src_planes = planes_of(&src);

    // Direction 1: independent 4444 encode → our decoder, scored against
    // ffmpeg's own decode of the identical packet.
    assert!(ffmpeg_ok(&[
        "-f",
        "lavfi",
        "-i",
        &lavfi,
        "-frames:v",
        "1",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuv444p10le",
        ff_mov.to_str().unwrap(),
    ]));
    assert!(ffmpeg_ok(&[
        "-i",
        ff_mov.to_str().unwrap(),
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuv444p10le",
        "-f",
        "rawvideo",
        ff_dec.to_str().unwrap(),
    ]));
    let ff_mov_bytes = std::fs::read(&ff_mov).expect("read ffmpeg mov");
    let ff_pkt = extract_prores_packet(&ff_mov_bytes);
    let ff_dec_raw = std::fs::read(&ff_dec).expect("read ffmpeg decode");
    let ff_planes = planes_of(&ff_dec_raw);
    let ours = decode_packet_with_options(
        &ff_pkt,
        None,
        Some((BitDepth::Ten, ChromaFormat::Y444)),
        OutputRange::Full,
    )
    .expect("decode ffmpeg's 4444 packet");
    let our_planes = ours.image_planes();
    assert_eq!(our_planes.len(), 3);
    for (p, name) in [(0usize, "Y"), (1, "Cb"), (2, "Cr")] {
        assert_eq!(our_planes[p].stride, (W * 2) as usize, "{name} stride");
        let psnr = psnr_le16(&our_planes[p].data, ff_planes[p], 0, 1023.0);
        eprintln!("ffmpeg 4444 → ours: {name} PSNR vs ffmpeg decode = {psnr:.2} dB");
        assert!(
            psnr >= MIN_CHROMA_PSNR_DB,
            "{name}: our decode of an independent 4444 packet is {psnr:.2} dB from \
             ffmpeg's (block order within the macroblock?)"
        );
    }

    // Direction 2: our 4444 encode → ffmpeg decode, scored against the
    // source. The ffmpeg MOV above doubles as the container template.
    // ffmpeg surfaces ap4h as yuv444p12le; it is compared against the
    // 10-bit source shifted up by two bits rather than asking ffmpeg to
    // down-convert (our header leaves matrix_coefficients at 0 =
    // unspecified, which ffmpeg's converter declines to rescale).
    let frame = VideoFrame {
        planes: vec![
            VideoPlane {
                data: src_planes[0].to_vec(),
                stride: (W * 2) as usize,
            },
            VideoPlane {
                data: src_planes[1].to_vec(),
                stride: (W * 2) as usize,
            },
            VideoPlane {
                data: src_planes[2].to_vec(),
                stride: (W * 2) as usize,
            },
        ],
        pts: Some(0),
    };
    let pkt = encode_frame_with_depth(
        &frame,
        W,
        H,
        ChromaFormat::Y444,
        BitDepth::Ten,
        Profile::Prores4444,
        2,
    )
    .expect("encode 4444 10-bit");
    std::fs::write(&our_mov, patch_mov_with_packet(&ff_mov_bytes, &pkt)).expect("write mov");
    assert!(
        ffmpeg_ok(&[
            "-i",
            our_mov.to_str().unwrap(),
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv444p12le",
            "-f",
            "rawvideo",
            our_dec.to_str().unwrap(),
        ]),
        "ffmpeg refused our 4444 packet"
    );
    let our_dec_raw = std::fs::read(&our_dec).expect("read ffmpeg decode of ours");
    let dec_planes = planes_of(&our_dec_raw);
    for (p, name) in [(0usize, "Y"), (1, "Cb"), (2, "Cr")] {
        let psnr = psnr_le16(dec_planes[p], src_planes[p], 2, 4095.0);
        eprintln!("ours 4444 → ffmpeg: {name} PSNR vs source = {psnr:.2} dB");
        assert!(
            psnr >= MIN_CHROMA_PSNR_DB,
            "{name}: ffmpeg's decode of our 4444 packet is {psnr:.2} dB from the source \
             (block order within the macroblock?)"
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
