//! Cross-decode acceptance test: bitstreams produced by THIS crate's
//! interlaced encoder must be decodable by ffmpeg's `prores_ks` decoder
//! at acceptable luma PSNR. The companion `ffmpeg_interop.rs` runs the
//! opposite direction (ffmpeg-encoded → our decoder); together they
//! close the interlaced-encode acceptance bar called out in the round
//! 92 dispatch.
//!
//! The test:
//!
//! 1. Generates a synthetic field-distinct 4:2:2 source whose even and
//!    odd rows differ in luma brightness (so a swapped-field bug shows
//!    up as a PSNR cliff).
//! 2. Encodes it via `encode_frame_interlaced(... interlace_mode=1|2)`.
//! 3. Wraps the raw `icpf`-prefixed packet in a minimal QuickTime MOV
//!    by substituting our packet into an ffmpeg-generated template MOV
//!    (same width/height/profile/pix_fmt/interlace_mode). The template
//!    MOV is used as an opaque container scaffold — only the `mdat`
//!    payload + `mdat`-size + `stsz` sample-size atoms are patched.
//! 4. Asks ffmpeg to decode the patched MOV to raw 10-bit YUV.
//! 5. Compares the decoded luma plane to the original source via PSNR.
//!
//! The wrapper-from-template trick avoids hand-rolling a full MOV
//! writer inside the prores crate (the umbrella's `oxideav-mov` exists
//! but cross-crate dev-deps are forbidden per the workspace rules).
//! All MOV-level knowledge consumed here comes from inspecting the
//! ffmpeg-produced container bytes at runtime, not from reading any
//! external library source.
//!
//! Skips gracefully when `ffmpeg` is missing.

use std::process::Command;

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::encode_frame_interlaced;
use oxideav_prores::frame::{ChromaFormat, Profile};

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tempdir() -> Option<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let p = base.join(format!("oxideav-prores-cross-{pid}-{ts}"));
    std::fs::create_dir_all(&p).ok()?;
    Some(p)
}

/// Build a synthetic field-distinct 4:2:2 8-bit frame. Even rows are
/// bright, odd rows are dim — a swapped TFF/BFF bug inverts the
/// pattern in the decoded output. A non-trivial in-row gradient +
/// chroma modulation stresses the entropy coder so the test catches
/// regressions in DC/AC code-tables (not just the field-pair plumbing).
fn synthetic_field_distinct(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![128u8; cw * h];
    let mut cr = vec![128u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            // Field-distinct brightness bias + an in-row diagonal
            // gradient + a per-row phase so the AC coefficients are
            // non-zero across most blocks.
            let base: i32 = if j % 2 == 0 { 160 } else { 96 };
            let grad = ((i + j) as u16 % 48) as i32;
            let phase = ((i.wrapping_mul(7) ^ j.wrapping_mul(13)) % 24) as i32;
            y[j * w + i] = (base + grad + phase - 12).clamp(16, 235) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] =
                (128 + ((i as i32 - cw as i32 / 2) * 2 + (j as i32 % 7)).clamp(-48, 48)) as u8;
            cr[j * cw + i] =
                (128 + ((j as i32 - h as i32 / 2) + (i as i32 % 5) * 2).clamp(-48, 48)) as u8;
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

/// Ask ffmpeg to produce a 1-frame interlaced ProRes MOV at the given
/// dimensions / profile / top-field-first. The resulting file is used
/// as a *template* — we will overwrite its `mdat` payload with our own
/// encoder output.
fn ffmpeg_make_template_mov(
    profile_flag: u8,
    width: u32,
    height: u32,
    top_field_first: bool,
    out_path: &std::path::Path,
) -> bool {
    let input = format!("testsrc=size={width}x{height}:rate=25:duration=1");
    let top_flag = if top_field_first { "1" } else { "0" };
    Command::new("ffmpeg")
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
            out_path.to_str().unwrap_or(""),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Find a top-level atom whose 4-byte name matches `name`. Returns the
/// offset of the size field (i.e. 4 bytes before the name) and the
/// size value. `start` lets the caller skip atoms already located.
fn find_top_atom(buf: &[u8], name: &[u8; 4], start: usize) -> Option<(usize, u32)> {
    let mut i = start;
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

/// Find the first 4-byte occurrence of `needle` inside `buf`. Linear
/// scan over the whole container (used to locate `stsz` / `stco`
/// which live deep inside `moov/trak/mdia/minf/stbl`).
fn find_atom_anywhere(buf: &[u8], needle: &[u8; 4]) -> Option<usize> {
    buf.windows(4).position(|w| w == needle)
}

/// Substitute our `icpf`-prefixed packet into a template MOV. Updates:
/// * the `mdat` atom's size field (4 bytes at `mdat_offset - 4`),
/// * the embedded `frame_size` at the start of the mdat payload,
/// * the first sample-size entry of the `stsz` atom (deep in `moov`).
///
/// The `stco` first-chunk-offset (file-offset of the mdat payload)
/// stays valid because the patched mdat begins at the same file
/// offset as the template's mdat. `moov` may shift in absolute file
/// position (forward or backward by the packet-size delta), but its
/// internal references to mdat are file-offset-relative and remain
/// correct.
fn patch_mov_with_packet(template: &[u8], pkt: &[u8]) -> Vec<u8> {
    let (mdat_off, mdat_size) =
        find_top_atom(template, b"mdat", 0).expect("template MOV must contain mdat");
    let mdat_payload_start = mdat_off + 8;
    let mdat_payload_end = mdat_off + mdat_size as usize;
    assert!(
        mdat_payload_end <= template.len(),
        "mdat extends past template length"
    );

    // Split template into head (up to mdat payload) and tail (atoms
    // after mdat). The patched file is head || pkt || tail with mdat
    // size + stsz sample size updated.
    let head_end = mdat_payload_start;
    let mut out = Vec::with_capacity(head_end + pkt.len() + (template.len() - mdat_payload_end));
    out.extend_from_slice(&template[..head_end]);
    out.extend_from_slice(pkt);
    out.extend_from_slice(&template[mdat_payload_end..]);

    // Update mdat atom size (8 byte atom header + payload).
    let new_mdat_size = (8 + pkt.len()) as u32;
    out[mdat_off..mdat_off + 4].copy_from_slice(&new_mdat_size.to_be_bytes());

    // Update stsz first-entry sample size. `stsz` layout (after the
    // 8-byte atom header):
    //   * 4 bytes version+flags
    //   * 4 bytes sample_size (0 means per-sample sizes follow)
    //   * 4 bytes sample_count
    //   * if sample_size == 0: sample_count * 4 bytes of entries
    //
    // For a 1-sample MOV ffmpeg's prores_ks emits a non-zero
    // `sample_size` directly (sample_count = 1); for multi-sample
    // files it would emit zero + a per-sample table. Either case is
    // handled below.
    let stsz_off = find_atom_anywhere(&out, b"stsz").expect("stsz");
    let sample_size_off = stsz_off + 8;
    let sample_count_off = stsz_off + 12;
    let sample_size = u32::from_be_bytes(
        out[sample_size_off..sample_size_off + 4]
            .try_into()
            .unwrap(),
    );
    let sample_count = u32::from_be_bytes(
        out[sample_count_off..sample_count_off + 4]
            .try_into()
            .unwrap(),
    );
    if sample_size != 0 {
        // Single uniform sample size — just rewrite the constant.
        out[sample_size_off..sample_size_off + 4]
            .copy_from_slice(&(pkt.len() as u32).to_be_bytes());
    } else {
        // Per-sample table follows. We only expect 1 sample in the
        // template; rewrite the first entry.
        assert!(sample_count >= 1, "stsz sample_count is zero");
        let first_entry_off = stsz_off + 16;
        out[first_entry_off..first_entry_off + 4]
            .copy_from_slice(&(pkt.len() as u32).to_be_bytes());
    }

    out
}

/// Convert an 8-bit luma plane to 10-bit LE (each sample << 2).
fn upshift_8_to_10_le(plane8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(plane8.len() * 2);
    for &v in plane8 {
        let v10 = (v as u16) << 2;
        out.extend_from_slice(&v10.to_le_bytes());
    }
    out
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

fn cross_decode_interlaced(profile: Profile, width: u32, height: u32, interlace_mode: u8) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping cross-decode test");
        return;
    }
    let tmp = tempdir().expect("tempdir");
    let tff = interlace_mode == 1;
    let profile_flag: u8 = match profile {
        Profile::Proxy => 0,
        Profile::Lt => 1,
        Profile::Standard => 2,
        Profile::Hq => 3,
        Profile::Prores4444 => 4,
        Profile::Prores4444Xq => 5,
    };
    let template_path = tmp.join(format!(
        "template_p{profile_flag}_{width}x{height}_im{interlace_mode}.mov"
    ));
    if !ffmpeg_make_template_mov(profile_flag, width, height, tff, &template_path) {
        eprintln!("template MOV unavailable — skipping (profile={profile_flag})");
        return;
    }
    let template = std::fs::read(&template_path).expect("read template");

    // Encode the same dimensions via our interlaced encoder.
    let src = synthetic_field_distinct(width, height);
    let pkt = encode_frame_interlaced(
        &src,
        width,
        height,
        ChromaFormat::Y422,
        BitDepth::Eight,
        profile,
        2, // qi = 2 → highest quality at HQ defaults
        None,
        interlace_mode,
    )
    .expect("encode_frame_interlaced");

    // Sanity: the encoded frame header should report the requested
    // interlace_mode + 2 pictures.
    let (fh, _) = oxideav_prores::frame::parse_frame(&pkt).expect("parse our packet");
    assert_eq!(
        fh.interlace_mode, interlace_mode,
        "our encoder's interlace_mode"
    );
    assert_eq!(
        fh.picture_count(),
        2,
        "interlaced frame must carry 2 pictures"
    );

    let patched = patch_mov_with_packet(&template, &pkt);
    let patched_path = tmp.join(format!(
        "patched_p{profile_flag}_{width}x{height}_im{interlace_mode}.mov"
    ));
    std::fs::write(&patched_path, &patched).expect("write patched");

    // Decode via ffmpeg to raw 10-bit YUV.
    let decoded_path = tmp.join("decoded.yuv");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            patched_path.to_str().unwrap(),
            "-pix_fmt",
            "yuv422p10le",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            decoded_path.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg decode");
    assert!(
        status.success(),
        "ffmpeg failed to decode our interlaced ProRes packet (profile={profile_flag}, im={interlace_mode})"
    );
    let decoded = std::fs::read(&decoded_path).expect("read decoded");
    let y_bytes_10 = (width as usize) * (height as usize) * 2;
    assert!(
        decoded.len() >= y_bytes_10,
        "ffmpeg produced {} bytes, expected at least {} for the luma plane",
        decoded.len(),
        y_bytes_10
    );
    let decoded_y = &decoded[..y_bytes_10];

    // Upshift our 8-bit source to 10-bit for an apples-to-apples PSNR
    // comparison against ffmpeg's 10-bit decoded output.
    let src_y_10 = upshift_8_to_10_le(&src.planes[0].data);
    let psnr = psnr_10bit(&src_y_10, decoded_y);
    eprintln!(
        "cross-decode interlaced (profile={profile_flag}, im={interlace_mode}, {width}x{height}): \
         packet={} bytes, luma PSNR={psnr:.2} dB",
        pkt.len()
    );
    // 30 dB at qi=2 is comfortably above the visual-quality knee for
    // an 8-bit→10-bit→ffmpeg-decode chain. A swapped-field bug would
    // dump PSNR below 10 dB; a broken DCT path collapses to ~6 dB.
    assert!(
        psnr >= 30.0,
        "interlaced cross-decode PSNR {psnr:.2} dB under 30 dB bar"
    );

    // Even-row vs odd-row luma sum check — defends against a TFF/BFF
    // mis-tag where ffmpeg interleaves the fields the wrong way.
    let mut even_sum = 0u64;
    let mut odd_sum = 0u64;
    let w = width as usize;
    for j in 0..(height as usize) {
        let row_start = j * w * 2;
        let mut row_sum = 0u64;
        for i in 0..w {
            let v = u16::from_le_bytes([
                decoded_y[row_start + i * 2],
                decoded_y[row_start + i * 2 + 1],
            ]);
            row_sum += v as u64;
        }
        if j % 2 == 0 {
            even_sum += row_sum;
        } else {
            odd_sum += row_sum;
        }
    }
    assert!(
        even_sum > odd_sum,
        "interlaced cross-decode: even-row sum {even_sum} not > odd-row sum {odd_sum} \
         (TFF/BFF field assignment swapped in our encoder?)"
    );
}

#[test]
fn cross_decode_apch_interlaced_tff() {
    cross_decode_interlaced(Profile::Hq, 64, 48, 1);
}

#[test]
fn cross_decode_apch_interlaced_bff() {
    cross_decode_interlaced(Profile::Hq, 64, 48, 2);
}

#[test]
fn cross_decode_apcn_interlaced_tff() {
    cross_decode_interlaced(Profile::Standard, 64, 48, 1);
}

#[test]
fn cross_decode_apcn_interlaced_bff() {
    cross_decode_interlaced(Profile::Standard, 64, 48, 2);
}

#[test]
fn cross_decode_apch_interlaced_larger() {
    // 128x96 — twice the macroblock grid; verifies the per-field
    // half-height picture path at a non-trivial resolution.
    cross_decode_interlaced(Profile::Hq, 128, 96, 1);
}
