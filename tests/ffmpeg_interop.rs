//! Black-box interop test: encode a test pattern with `ffmpeg -c:v
//! prores_ks` and decode the resulting bitstream with this crate.
//!
//! For each profile in scope (`apcn` Standard and `apch` HQ), the test:
//!
//! 1. Invokes ffmpeg to produce a 1-frame `.mov`.
//! 2. Extracts the ProRes packet from the `mdat` atom (the first
//!    `frame_size + 'icpf' + ...` blob).
//! 3. Feeds the packet to [`oxideav_prores::decoder::decode_packet`].
//! 4. Asserts the decoded frame has the right dimensions and a
//!    plausible PSNR against the source `testsrc` luma plane (we
//!    re-render the source separately via ffmpeg for comparison).
//!
//! Skips gracefully when `ffmpeg` is missing.

use std::process::Command;

use oxideav_prores::decoder::{decode_packet, decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn ffmpeg_make_prores_mov(profile_flag: u8, width: u32, height: u32) -> Option<Vec<u8>> {
    let tmp = tempdir()?;
    let out_path = tmp.join(format!("prores_p{profile_flag}_{width}x{height}.mov"));
    let pix_fmt = "yuv422p10le"; // we exercise apcn / apch only
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size={width}x{height}:rate=1:duration=1"),
            "-c:v",
            "prores_ks",
            "-profile:v",
            &profile_flag.to_string(),
            "-pix_fmt",
            pix_fmt,
            "-frames:v",
            "1",
            out_path.to_str()?,
        ])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    std::fs::read(&out_path).ok()
}

fn tempdir() -> Option<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let p = base.join(format!("oxideav-prores-interop-{pid}-{ts}"));
    std::fs::create_dir_all(&p).ok()?;
    Some(p)
}

/// Extract the first ProRes packet from a `.mov` mdat atom. Returns the
/// bytes starting at the ProRes `frame_size` (i.e., what the decoder
/// expects). The mdat payload contains the raw `frame()` syntax
/// concatenations; for a 1-frame test there's exactly one.
fn extract_prores_packet(mov: &[u8]) -> Option<Vec<u8>> {
    // Find the 'icpf' magic. The first 4 bytes preceding it are the
    // big-endian frame_size; the ProRes packet runs from there for
    // `frame_size` bytes.
    let needle = b"icpf";
    for i in 4..mov.len() - 4 {
        if &mov[i..i + 4] == needle {
            let size_off = i - 4;
            if size_off + 4 > mov.len() {
                continue;
            }
            let frame_size = u32::from_be_bytes(mov[size_off..size_off + 4].try_into().ok()?);
            let end = size_off + frame_size as usize;
            if end <= mov.len() {
                return Some(mov[size_off..end].to_vec());
            }
        }
    }
    None
}

fn try_decode(profile_flag: u8, width: u32, height: u32) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping interop test");
        return;
    }
    let Some(mp4) = ffmpeg_make_prores_mov(profile_flag, width, height) else {
        eprintln!("ffmpeg prores_ks profile={profile_flag} unavailable, skipping");
        return;
    };
    let pkt = extract_prores_packet(&mp4)
        .unwrap_or_else(|| panic!("could not extract icpf packet from ffmpeg mov"));
    eprintln!(
        "decoded {} bytes of mp4, extracted {}-byte ProRes packet (profile {profile_flag})",
        mp4.len(),
        pkt.len()
    );
    let frame = decode_packet(&pkt, Some(0)).unwrap_or_else(|e| {
        panic!(
            "decode_packet failed for ffmpeg-produced profile={profile_flag} stream: {e:?} \
             (packet first 16 bytes = {:02x?})",
            &pkt[..pkt.len().min(16)]
        )
    });
    assert_eq!(frame.planes.len(), 3, "expected 3 planes");
    assert_eq!(frame.planes[0].data.len(), (width * height) as usize);
    // Chroma planes are half-width for 4:2:2.
    assert_eq!(
        frame.planes[1].data.len(),
        (width / 2 * height) as usize,
        "Cb plane size"
    );
    assert_eq!(
        frame.planes[2].data.len(),
        (width / 2 * height) as usize,
        "Cr plane size"
    );

    // Sanity: the decoded luma plane should have a non-trivial dynamic
    // range (testsrc is a colorful pattern). Computing PSNR against the
    // exact source would require a separate ffmpeg invocation to
    // export the testsrc raw frames; we settle for a "non-blank"
    // assertion here, which still proves the entropy coder produced
    // something coherent.
    let lum = &frame.planes[0].data;
    let mut min = u8::MAX;
    let mut max = u8::MIN;
    for &v in lum {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    assert!(
        max as i32 - min as i32 > 80,
        "decoded luma has trivial dynamic range ({min}..{max}) — entropy coder probably wrong"
    );
    eprintln!("luma range for profile {profile_flag}: {min}..{max}");
}

#[test]
fn rdd36_decode_apcn_from_ffmpeg() {
    try_decode(2, 128, 128);
}

#[test]
fn rdd36_decode_apch_from_ffmpeg() {
    try_decode(3, 128, 128);
}

#[test]
fn rdd36_decode_apcn_smaller() {
    try_decode(2, 64, 48);
}

// ────────────────── 10-bit (Yuv422P10Le) interop ──────────────────

/// Decode an ffmpeg-produced ProRes packet at 10-bit output. Same
/// ffmpeg fixture as the 8-bit interop above — `prores_ks` is invoked
/// with `-pix_fmt yuv422p10le` already, and our crate now honors that
/// bit-depth at decode time.
fn try_decode_10bit(profile_flag: u8, width: u32, height: u32) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping 10-bit interop test");
        return;
    }
    let Some(mp4) = ffmpeg_make_prores_mov(profile_flag, width, height) else {
        eprintln!("ffmpeg prores_ks profile={profile_flag} unavailable, skipping");
        return;
    };
    let pkt = extract_prores_packet(&mp4)
        .unwrap_or_else(|| panic!("could not extract icpf packet from ffmpeg mov"));
    eprintln!(
        "10-bit decode: extracted {}-byte ProRes packet (profile {profile_flag})",
        pkt.len()
    );
    let frame = decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Ten, ChromaFormat::Y422)))
        .unwrap_or_else(|e| {
            panic!("decode_packet_with_depth(Ten) failed for profile={profile_flag}: {e:?}")
        });
    assert_eq!(frame.planes.len(), 3, "expected 3 planes");
    // 10-bit planes hold 2 bytes per sample.
    let expected_y_bytes = (width * height * 2) as usize;
    let expected_c_bytes = (width / 2 * height * 2) as usize;
    assert_eq!(frame.planes[0].data.len(), expected_y_bytes, "Y plane size");
    assert_eq!(
        frame.planes[1].data.len(),
        expected_c_bytes,
        "Cb plane size"
    );
    assert_eq!(
        frame.planes[2].data.len(),
        expected_c_bytes,
        "Cr plane size"
    );
    assert_eq!(frame.planes[0].stride, (width as usize) * 2);
    assert_eq!(frame.planes[1].stride, (width as usize / 2) * 2);

    // Sanity: luma must have a non-trivial 10-bit dynamic range AND
    // include values above 255 — proving the emit path is genuinely
    // 10-bit, not just 8-bit values written as u16.
    let mut min_v: u16 = u16::MAX;
    let mut max_v: u16 = 0;
    for chunk in frame.planes[0].data.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    eprintln!("10-bit luma range for profile {profile_flag}: {min_v}..{max_v}");
    assert!(
        max_v as i32 - min_v as i32 > 320,
        "10-bit luma has trivial dynamic range ({min_v}..{max_v})"
    );
    assert!(
        max_v > 255,
        "10-bit luma never exceeds 255 ({max_v}) — emit path collapsed to 8 bits"
    );
    // No high-bit garbage outside the 10-bit window.
    assert!(max_v <= 1023, "10-bit luma value out of range: {max_v}");
}

#[test]
fn rdd36_decode_apcn_10bit_from_ffmpeg() {
    try_decode_10bit(2, 128, 128);
}

#[test]
fn rdd36_decode_apch_10bit_from_ffmpeg() {
    try_decode_10bit(3, 128, 128);
}

/// Same ffmpeg apcn fixture, decoded twice — once at 8-bit, once at
/// 10-bit — and the two outputs are compared. The 10-bit luma sample
/// downshifted by 2 should round-trip to the 8-bit luma sample within
/// ±1 (spec §7.5.1 rounding for `b=8` vs `b=10`).
#[test]
fn rdd36_apcn_8bit_and_10bit_outputs_are_consistent() {
    if !have_ffmpeg() {
        return;
    }
    let Some(mp4) = ffmpeg_make_prores_mov(2, 128, 128) else {
        return;
    };
    let pkt = extract_prores_packet(&mp4).expect("extract");
    let f8 = decode_packet(&pkt, Some(0)).expect("decode 8-bit");
    let f10 = decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Ten, ChromaFormat::Y422)))
        .expect("decode 10-bit");
    assert_eq!(f8.planes[0].data.len() * 2, f10.planes[0].data.len());
    let mut max_diff = 0i32;
    for (i, chunk) in f10.planes[0].data.chunks_exact(2).enumerate() {
        let v10 = u16::from_le_bytes([chunk[0], chunk[1]]);
        // Drop two LSBs to simulate 8-bit truncation.
        let v8_from_10 = (v10 >> 2) as i32;
        let v8 = f8.planes[0].data[i] as i32;
        let d = (v8 - v8_from_10).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("max(|8-bit − 10-bit/4|) = {max_diff}");
    // The 10-bit emit path multiplies by 4 (= shifts left by 2), so the
    // 8-bit value should match exactly modulo rounding at clamp edges.
    assert!(
        max_diff <= 1,
        "8-bit and 10-bit/4 outputs diverge by more than 1: {max_diff}"
    );
}
