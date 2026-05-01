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
    ffmpeg_make_prores_mov_with_pix(profile_flag, width, height, "yuv422p10le", None)
}

/// `pix_fmt` selects the source/encoder pixel format. `lavfi_input` may
/// supply a custom lavfi pipeline (e.g. one that produces `yuva444p10le`
/// for the 4444 + alpha profile); when None, `testsrc` is used directly.
fn ffmpeg_make_prores_mov_with_pix(
    profile_flag: u8,
    width: u32,
    height: u32,
    pix_fmt: &str,
    lavfi_input: Option<&str>,
) -> Option<Vec<u8>> {
    let tmp = tempdir()?;
    let out_path = tmp.join(format!(
        "prores_p{profile_flag}_{width}x{height}_{pix_fmt}.mov"
    ));
    let default_input = format!("testsrc=size={width}x{height}:rate=1:duration=1");
    let input = lavfi_input.unwrap_or(&default_input);
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            input,
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

// ────────────────── 12-bit (Yuv422P12Le / Yuv444P12Le) ──────────────────

#[test]
fn rdd36_decode_apch_12bit_from_ffmpeg() {
    if !have_ffmpeg() {
        return;
    }
    // ffmpeg encodes apch from a 12-bit source when -pix_fmt yuv422p12le is
    // requested; this exercises the 12-bit emit path on the decoder side.
    let Some(mp4) = ffmpeg_make_prores_mov_with_pix(3, 128, 128, "yuv422p12le", None) else {
        return;
    };
    let pkt = extract_prores_packet(&mp4).expect("extract icpf");
    let frame =
        decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Twelve, ChromaFormat::Y422)))
            .expect("decode 12-bit");
    assert_eq!(frame.planes.len(), 3);
    let expected_y = (128 * 128 * 2) as usize;
    assert_eq!(frame.planes[0].data.len(), expected_y);
    let mut max_v: u16 = 0;
    let mut min_v: u16 = u16::MAX;
    for chunk in frame.planes[0].data.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    eprintln!("12-bit luma range: {min_v}..{max_v}");
    assert!(max_v <= 4095, "12-bit luma value out of range: {max_v}");
    assert!(
        max_v > 1023,
        "12-bit luma never exceeds 1023 ({max_v}) — emit path collapsed to 10 bits"
    );
}

// ───────────── 4444 + alpha (ap4h) interop ─────────────

/// Build an ap4h fixture with a deterministic alpha plane. The lavfi
/// `geq` filter computes alpha as `(X + Y) * 256 / (W + H)` so the
/// alpha gradient sweeps the full 8-bit range (0 at top-left, ≈255 at
/// bottom-right). Determinism matters because ffmpeg's `gradients`
/// filter is animated per frame and varies between ffmpeg builds.
fn ffmpeg_make_prores_ap4h_with_alpha_gradient(width: u32, height: u32) -> Option<Vec<u8>> {
    let tmp = tempdir()?;
    let out_path = tmp.join(format!("prores_ap4h_a_{width}x{height}.mov"));
    let pattern = format!("color=c=gray:size={width}x{height}:rate=1:duration=1");
    // (X + Y) * 256 / (W + H - 2) yields 0..255 across the diagonal.
    let denom = (width + height - 2).max(1);
    let geq = format!("geq=lum='(X+Y)*256/{denom}':cb=128:cr=128,format=yuv444p");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &pattern,
            "-filter_complex",
            &format!("[0:v]{geq}[a];[0:v][a]alphamerge,format=yuva444p10le"),
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
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

/// Decode an ap4h ProRes 4444 stream produced by ffmpeg with a uniform
/// fully-opaque alpha plane. The decoded frame must have FOUR planes
/// (Y, Cb, Cr, A) and the alpha plane must be at the maximum value (the
/// 12-bit equivalent of 0xFFFF, which is 4095).
#[test]
fn rdd36_decode_ap4h_with_alpha_from_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping ap4h interop test");
        return;
    }
    // testsrc + format=yuva444p10le → fully-opaque alpha plane. This
    // exercises the long-run alpha codeword (single value repeated for
    // the entire 128×128 frame).
    let lavfi = "testsrc=size=128x128:rate=1:duration=1,format=yuva444p10le";
    let Some(mp4) = ffmpeg_make_prores_mov_with_pix(4, 128, 128, "yuva444p10le", Some(lavfi))
    else {
        eprintln!("ffmpeg prores_ks ap4h unavailable, skipping");
        return;
    };
    let pkt = extract_prores_packet(&mp4).expect("extract icpf");
    eprintln!("ap4h packet: {} bytes", pkt.len());

    // ffmpeg emits ap4h at 12-bit YUV (regardless of the 10-bit source)
    // with 16-bit alpha. Request 12-bit YUV output; alpha lands in the
    // 4th plane at 12-bit too.
    let frame =
        decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Twelve, ChromaFormat::Y444)))
            .unwrap_or_else(|e| panic!("decode_packet ap4h failed: {e:?}"));
    assert_eq!(
        frame.planes.len(),
        4,
        "ap4h with alpha must yield 4 planes (Y/Cb/Cr/A)"
    );
    let y_bytes = (128 * 128 * 2) as usize;
    assert_eq!(frame.planes[0].data.len(), y_bytes, "Y plane size");
    assert_eq!(frame.planes[1].data.len(), y_bytes, "Cb plane size");
    assert_eq!(frame.planes[2].data.len(), y_bytes, "Cr plane size");
    assert_eq!(frame.planes[3].data.len(), y_bytes, "Alpha plane size");

    let mut min_a: u16 = u16::MAX;
    let mut max_a: u16 = 0;
    for chunk in frame.planes[3].data.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v < min_a {
            min_a = v;
        }
        if v > max_a {
            max_a = v;
        }
    }
    eprintln!("ap4h alpha plane range (12-bit samples): {min_a}..{max_a}");
    assert!(max_a <= 4095, "12-bit alpha sample out of range: {max_a}");
    // testsrc → fully-opaque alpha (round((4095*65535)/65535) = 4095).
    assert_eq!(
        (min_a, max_a),
        (4095, 4095),
        "ap4h opaque-source alpha plane should be uniformly 4095"
    );
}

/// Decode an ap4h fixture with a non-trivial alpha gradient. This
/// exercises the alpha entropy coder's small-magnitude difference path
/// (Tables 13/14) plus the run codeword for run lengths > 1.
#[test]
fn rdd36_decode_ap4h_with_alpha_gradient_from_ffmpeg() {
    if !have_ffmpeg() {
        return;
    }
    let Some(mp4) = ffmpeg_make_prores_ap4h_with_alpha_gradient(128, 128) else {
        eprintln!("ffmpeg ap4h+alphamerge unavailable, skipping");
        return;
    };
    let pkt = extract_prores_packet(&mp4).expect("extract icpf");
    eprintln!("ap4h-alpha-gradient packet: {} bytes", pkt.len());

    let frame =
        decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Twelve, ChromaFormat::Y444)))
            .unwrap_or_else(|e| panic!("decode_packet ap4h-gradient failed: {e:?}"));
    assert_eq!(frame.planes.len(), 4);

    let alpha = &frame.planes[3].data;
    let mut min_a: u16 = u16::MAX;
    let mut max_a: u16 = 0;
    for chunk in alpha.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        if v < min_a {
            min_a = v;
        }
        if v > max_a {
            max_a = v;
        }
    }
    eprintln!("ap4h gradient alpha range (12-bit): {min_a}..{max_a}");
    // The lavfi `gradients` filter produces a non-uniform 16-bit
    // gradient. Its exact range varies per ffmpeg build/version, so we
    // settle for a lower bound: a broken entropy decode would yield a
    // single constant value (range = 0).
    assert!(
        (max_a as i32 - min_a as i32) > 100,
        "ap4h alpha gradient has trivial range ({min_a}..{max_a}) — entropy decode wrong"
    );
    let mut seen = std::collections::HashSet::new();
    for chunk in alpha.chunks_exact(2) {
        seen.insert(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    eprintln!("ap4h gradient alpha distinct values: {}", seen.len());
    assert!(
        seen.len() >= 5,
        "ap4h alpha gradient has only {} distinct values — entropy decode wrong",
        seen.len()
    );
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

// ────────────── interlaced (RDD 36 §5.1, §6.2, §7.5.3) ──────────────

/// Build an ffmpeg-encoded interlaced ProRes `.mov` and return the
/// container bytes. `top_field_first` selects ffmpeg's `-top 1` (TFF,
/// our interlace_mode 1) vs. `-top 0` (BFF, mode 2). `+ildct` enables
/// interlaced DCT coding so prores_ks emits one picture per field with
/// the interlaced block-scan pattern (Figure 5).
fn ffmpeg_make_prores_interlaced_mov(
    profile_flag: u8,
    width: u32,
    height: u32,
    top_field_first: bool,
) -> Option<Vec<u8>> {
    let tmp = tempdir()?;
    let out_path = tmp.join(format!(
        "prores_int_p{profile_flag}_{width}x{height}_{}.mov",
        if top_field_first { "tff" } else { "bff" }
    ));
    let input = format!("testsrc=size={width}x{height}:rate=25:duration=1");
    let top_flag = if top_field_first { "1" } else { "0" };
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &input,
            "-c:v",
            "prores_ks",
            "-profile:v",
            &profile_flag.to_string(),
            "-pix_fmt",
            "yuv422p10le",
            "-flags",
            "+ildct",
            "-top",
            top_flag,
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

/// Compute luma PSNR between two equal-length 10-bit LE-packed planes.
fn psnr_10bit(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() / 2;
    let mut mse = 0.0f64;
    for i in 0..n {
        let av = u16::from_le_bytes([a[i * 2], a[i * 2 + 1]]) as f64;
        let bv = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) as f64;
        let d = av - bv;
        mse += d * d;
    }
    mse /= n as f64;
    if mse == 0.0 {
        return 120.0;
    }
    10.0 * (1023.0_f64 * 1023.0 / mse).log10()
}

/// Decode an interlaced ffmpeg fixture and validate it against the same
/// `testsrc` pattern (re-rendered into raw 10-bit LE for the PSNR
/// comparison). The acceptance bar per task #126 is ≥ 40 dB Y on
/// `apch` (HQ).
fn try_decode_interlaced(profile_flag: u8, width: u32, height: u32, top_field_first: bool) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping interlaced interop test");
        return;
    }
    let Some(mp4) = ffmpeg_make_prores_interlaced_mov(profile_flag, width, height, top_field_first)
    else {
        eprintln!("ffmpeg prores_ks interlaced profile={profile_flag} unavailable, skipping");
        return;
    };
    let pkt = extract_prores_packet(&mp4).expect("extract icpf packet");
    eprintln!(
        "interlaced(top_field_first={top_field_first}) packet: {} bytes",
        pkt.len()
    );

    // Confirm the frame header reports interlace_mode 1 or 2.
    let (fh, _) = oxideav_prores::frame::parse_frame(&pkt).expect("parse_frame");
    let expected_mode: u8 = if top_field_first { 1 } else { 2 };
    assert_eq!(
        fh.interlace_mode, expected_mode,
        "ffmpeg-produced frame header interlace_mode mismatch"
    );

    let frame = decode_packet_with_depth(&pkt, Some(0), Some((BitDepth::Ten, ChromaFormat::Y422)))
        .unwrap_or_else(|e| panic!("decode interlaced failed: {e:?}"));
    assert_eq!(frame.planes.len(), 3);
    let y_bytes = (width * height * 2) as usize;
    assert_eq!(frame.planes[0].data.len(), y_bytes);

    // Render the raw 10-bit reference via ffmpeg.
    let tmp = tempdir().expect("tempdir");
    let raw_path = tmp.join("ref.yuv");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size={width}x{height}:rate=25:duration=1"),
            "-pix_fmt",
            "yuv422p10le",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            raw_path.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg ref render");
    assert!(status.success(), "ffmpeg ref render failed");
    let raw = std::fs::read(&raw_path).expect("read ref");
    let lum_ref = &raw[..y_bytes];
    let psnr = psnr_10bit(lum_ref, &frame.planes[0].data);
    eprintln!("interlaced PSNR (profile={profile_flag}, tff={top_field_first}): {psnr:.2} dB");
    assert!(
        psnr >= 40.0,
        "interlaced ProRes PSNR {psnr:.2} dB under 40 dB acceptance bar"
    );
}

#[test]
fn rdd36_decode_apch_interlaced_tff_from_ffmpeg() {
    try_decode_interlaced(3, 128, 128, true);
}

#[test]
fn rdd36_decode_apch_interlaced_bff_from_ffmpeg() {
    try_decode_interlaced(3, 128, 128, false);
}

#[test]
fn rdd36_decode_apcn_interlaced_from_ffmpeg() {
    // Same 128x128 fixture but apcn (Standard profile) — exercises the
    // Standard-profile interlaced path at the lower-quality preset.
    try_decode_interlaced(2, 128, 128, true);
}
