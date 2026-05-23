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
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::{encode_frame_interlaced, encode_frame_with_alpha};
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

/// Build a synthetic field-distinct 4:2:2 **10-bit** frame. Same
/// field-distinct structure as [`synthetic_field_distinct`] but the
/// samples are genuine 10-bit values (`0..=1023`) packed little-endian,
/// so the cross-decode exercises the HBD `read_sample` path
/// (RDD 36 §7.5.1: `v = s / 2^(b-9) - 256` for `b = 10`) combined with
/// the interlaced field-pair split (§7.5.3) — not an 8-bit value merely
/// shifted into 10-bit storage. `stride` is in **bytes** (2 per sample).
fn synthetic_field_distinct_10(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    // 10-bit SMPTE-legal-ish luma window (≈64..940) so the bright/dim
    // field bias survives ffmpeg's full-range clamp on decode.
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; cw * h * 2];
    let mut cr = vec![0u8; cw * h * 2];
    let put = |buf: &mut [u8], idx: usize, v: u16| {
        let off = idx * 2;
        buf[off] = (v & 0xFF) as u8;
        buf[off + 1] = (v >> 8) as u8;
    };
    for j in 0..h {
        for i in 0..w {
            // Field-distinct brightness bias scaled into 10-bit, plus an
            // in-row diagonal gradient + per-row phase so the AC
            // coefficients are non-zero across most blocks.
            let base: i32 = if j % 2 == 0 { 640 } else { 384 };
            let grad = ((i + j) as u16 % 192) as i32;
            let phase = ((i.wrapping_mul(7) ^ j.wrapping_mul(13)) % 96) as i32;
            let v = (base + grad + phase - 48).clamp(64, 940) as u16;
            put(&mut y, j * w + i, v);
        }
        for i in 0..cw {
            let cbv = (512 + ((i as i32 - cw as i32 / 2) * 8 + (j as i32 % 7) * 4).clamp(-192, 192))
                .clamp(64, 960) as u16;
            let crv = (512 + ((j as i32 - h as i32 / 2) * 4 + (i as i32 % 5) * 8).clamp(-192, 192))
                .clamp(64, 960) as u16;
            put(&mut cb, j * cw + i, cbv);
            put(&mut cr, j * cw + i, crv);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane {
                stride: cw * 2,
                data: cb,
            },
            VideoPlane {
                stride: cw * 2,
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
    cross_decode_interlaced_depth(profile, width, height, interlace_mode, BitDepth::Eight);
}

fn cross_decode_interlaced_depth(
    profile: Profile,
    width: u32,
    height: u32,
    interlace_mode: u8,
    bit_depth: BitDepth,
) {
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

    // Encode the same dimensions via our interlaced encoder. The 10-bit
    // path builds a genuine 10-bit source (LE u16 samples) so the HBD
    // field-pair packing — `read_sample`'s `BitDepth::Ten` branch
    // combined with the §7.5.3 field deinterleave — is what gets
    // validated, not an 8-bit value zero-padded into 10-bit storage.
    let src = match bit_depth {
        BitDepth::Eight => synthetic_field_distinct(width, height),
        BitDepth::Ten => synthetic_field_distinct_10(width, height),
        BitDepth::Twelve => unreachable!("12-bit not exercised here"),
    };
    let pkt = encode_frame_interlaced(
        &src,
        width,
        height,
        ChromaFormat::Y422,
        bit_depth,
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

    // Build the 10-bit reference luma plane. For the 8-bit source we
    // upshift each sample << 2; the 10-bit source is already 10-bit LE
    // so we compare it directly (apples-to-apples against ffmpeg's
    // yuv422p10le decode output).
    let src_y_10 = match bit_depth {
        BitDepth::Eight => upshift_8_to_10_le(&src.planes[0].data),
        BitDepth::Ten => src.planes[0].data.clone(),
        BitDepth::Twelve => unreachable!("12-bit not exercised here"),
    };
    let psnr = psnr_10bit(&src_y_10, decoded_y);
    eprintln!(
        "cross-decode interlaced (profile={profile_flag}, im={interlace_mode}, {width}x{height}, \
         {}-bit): packet={} bytes, luma PSNR={psnr:.2} dB",
        bit_depth.bits(),
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

// ───────────── 10-bit interlaced (HBD field-pair packing) ─────────────
//
// These exercise `encode_frame_interlaced` with `BitDepth::Ten`: the
// genuine-10-bit `read_sample` path (RDD 36 §7.5.1 level shift for
// b = 10) feeding the two-field §7.5.3 deinterleave. ffmpeg's
// `prores_ks` decoder must reconstruct the field-pair to ≥ 30 dB luma
// PSNR against the 10-bit source, and the bright-even / dim-odd field
// bias must survive (catches a swapped TFF/BFF field-pair tag in the
// HBD path specifically). The 8-bit cases above don't cover the
// `BitDepth::Ten` sample-read branch.

#[test]
fn cross_decode_apch_interlaced_10bit_tff() {
    cross_decode_interlaced_depth(Profile::Hq, 64, 48, 1, BitDepth::Ten);
}

#[test]
fn cross_decode_apch_interlaced_10bit_bff() {
    cross_decode_interlaced_depth(Profile::Hq, 64, 48, 2, BitDepth::Ten);
}

#[test]
fn cross_decode_apcn_interlaced_10bit_tff() {
    cross_decode_interlaced_depth(Profile::Standard, 64, 48, 1, BitDepth::Ten);
}

#[test]
fn cross_decode_apch_interlaced_10bit_larger() {
    // 128x96 10-bit — twice the macroblock grid in the HBD field path.
    cross_decode_interlaced_depth(Profile::Hq, 128, 96, 1, BitDepth::Ten);
}

// ─────────── interlaced 4444 + alpha (ap4h / ap4x field-pair) ───────────
//
// These exercise the hardest path the encoder owns simultaneously:
//   * 4:4:4 chroma (ChromaFormat::Y444 — full-resolution Cb/Cr blocks),
//   * genuine 12-bit Y'CbCr samples (RDD 36 §7.5.1 level shift for b = 12,
//     `read_sample`'s `BitDepth::Twelve` branch) — ffmpeg's ap4h/ap4x is
//     internally 12-bit, so this matches the decoder's native depth,
//   * a per-pixel **16-bit alpha** plane coded losslessly per RDD 36
//     §5.3.3 + §7.1.2 (raster-scan run/diff VLC, Table 14) at the tail of
//     every slice — emitted for the full padded MB-row height (§7.5.2),
//   * the §7.5.3 two-field deinterleave: the source plane is split into
//     top (rows 0,2,4,…) and bottom (rows 1,3,5,…) field pictures that
//     share one frame_header and each use the §7.2 Figure 5 interlaced
//     block scan.
//
// Earlier cross-decode cases cover interlaced 4:2:2 (8-bit and 10-bit) but
// none combines 4:4:4 + 12-bit + alpha + the field-pair split. ffmpeg's
// `prores_ks` decoder must reconstruct the field-pair to ≥ 30 dB luma PSNR
// against the 12-bit source AND recover the alpha gradient (a swapped
// TFF/BFF tag or a broken alpha-blob offset both show up as failures).

/// Build a synthetic field-distinct 4:4:4 **12-bit** frame with a
/// per-pixel **16-bit alpha** gradient. Even rows are bright, odd rows
/// are dim (a swapped TFF/BFF bug inverts the pattern); the alpha sweeps
/// the full 16-bit range diagonally so the alpha entropy coder's run +
/// difference codewords (Table 14) are exercised, not just a flat
/// constant. Y/Cb/Cr strides are in **bytes** (2 per 12-bit sample);
/// the alpha plane is 16-bit LE (2 bytes per sample).
fn synthetic_field_distinct_444_alpha_12(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; w * h * 2];
    let mut cr = vec![0u8; w * h * 2];
    let mut a = vec![0u8; w * h * 2];
    let put = |buf: &mut [u8], idx: usize, v: u16| {
        let off = idx * 2;
        buf[off] = (v & 0xFF) as u8;
        buf[off + 1] = (v >> 8) as u8;
    };
    for j in 0..h {
        for i in 0..w {
            // Field-distinct brightness bias scaled into 12-bit (≈256..3760
            // SMPTE-legal-ish window so the bias survives ffmpeg's clamp),
            // plus an in-row gradient + per-row phase so AC coefficients
            // are non-zero across most blocks.
            let base: i32 = if j % 2 == 0 { 2560 } else { 1536 };
            let grad = ((i + j) as u16 % 768) as i32;
            let phase = ((i.wrapping_mul(7) ^ j.wrapping_mul(13)) % 384) as i32;
            let v = (base + grad + phase - 192).clamp(256, 3760) as u16;
            put(&mut y, j * w + i, v);
            // Full-resolution 4:4:4 chroma modulation.
            let cbv = (2048
                + ((i as i32 - w as i32 / 2) * 16 + (j as i32 % 7) * 8).clamp(-768, 768))
            .clamp(256, 3840) as u16;
            let crv = (2048
                + ((j as i32 - h as i32 / 2) * 16 + (i as i32 % 5) * 8).clamp(-768, 768))
            .clamp(256, 3840) as u16;
            put(&mut cb, j * w + i, cbv);
            put(&mut cr, j * w + i, crv);
            // 16-bit alpha diagonal gradient: 0 at (0,0) → ~65535 at the
            // far corner. `.max(1)` avoids a degenerate all-zero corner.
            let av = (((i + j) * 65535 / (w + h)) as u16).max(1);
            put(&mut a, j * w + i, av);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane {
                stride: w * 2,
                data: cb,
            },
            VideoPlane {
                stride: w * 2,
                data: cr,
            },
            VideoPlane {
                stride: w * 2,
                data: a,
            },
        ],
    }
}

/// Ask ffmpeg to produce a 1-frame interlaced ProRes 4444/4444 XQ MOV
/// **with an alpha plane** (`yuva444p12le`). Used as a template scaffold
/// whose `mdat` payload we overwrite with our own encoder output. The
/// `format=yuva444p12le` filter forces ffmpeg to allocate the alpha plane
/// (and hence emit `alpha_channel_type != 0` in the sample-entry / header
/// scaffold), so the substituted packet lands in an alpha-aware container.
fn ffmpeg_make_template_mov_444_alpha(
    profile_flag: u8,
    width: u32,
    height: u32,
    top_field_first: bool,
    out_path: &std::path::Path,
) -> bool {
    let input = format!("testsrc=size={width}x{height}:rate=25:duration=1,format=yuva444p12le");
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
            "yuva444p12le",
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

/// Compute PSNR between two equal-length 12-bit LE-packed planes.
fn psnr_12bit(a: &[u8], b: &[u8]) -> f64 {
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
    10.0 * (4095.0_f64 * 4095.0 / mse).log10()
}

fn cross_decode_interlaced_4444_alpha(
    profile: Profile,
    width: u32,
    height: u32,
    interlace_mode: u8,
) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping 4444+alpha interlaced cross-decode");
        return;
    }
    assert!(
        matches!(profile, Profile::Prores4444 | Profile::Prores4444Xq),
        "this driver is for the 4444 / 4444 XQ profiles only"
    );
    let tmp = tempdir().expect("tempdir");
    let tff = interlace_mode == 1;
    let profile_flag: u8 = match profile {
        Profile::Prores4444 => 4,
        Profile::Prores4444Xq => 5,
        _ => unreachable!("non-4444 profile in 4444 driver"),
    };
    let template_path = tmp.join(format!(
        "template_a_p{profile_flag}_{width}x{height}_im{interlace_mode}.mov"
    ));
    if !ffmpeg_make_template_mov_444_alpha(profile_flag, width, height, tff, &template_path) {
        eprintln!("4444+alpha template MOV unavailable — skipping (profile={profile_flag})");
        return;
    }
    let template = std::fs::read(&template_path).expect("read template");

    // Encode the same dimensions via our interlaced encoder at genuine
    // 12-bit 4:4:4 with a 16-bit alpha plane. This drives:
    //   * `read_sample`'s `BitDepth::Twelve` branch (§7.5.1 level shift),
    //   * the §7.1.2 / Table 14 16-bit-alpha entropy coder (per-slice
    //     scanned-alpha blob at the padded MB-row height, §7.5.2),
    //   * the §7.5.3 two-field deinterleave (rows {0,2,…} top / {1,3,…}
    //     bottom).
    let src = synthetic_field_distinct_444_alpha_12(width, height);
    let pkt = encode_frame_interlaced(
        &src,
        width,
        height,
        ChromaFormat::Y444,
        BitDepth::Twelve,
        profile,
        2, // qi = 2 → finest at 4444 defaults
        Some(AlphaChannelType::Sixteen),
        interlace_mode,
    )
    .expect("encode_frame_interlaced 4444+alpha");

    // Sanity: the encoded frame header reports the requested
    // interlace_mode, 2 pictures, AND a non-zero alpha_channel_type.
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
    assert_eq!(
        fh.alpha_channel_type, 2,
        "4444+alpha frame must report alpha_channel_type=2 (16-bit)"
    );

    let patched = patch_mov_with_packet(&template, &pkt);
    let patched_path = tmp.join(format!(
        "patched_a_p{profile_flag}_{width}x{height}_im{interlace_mode}.mov"
    ));
    std::fs::write(&patched_path, &patched).expect("write patched");

    // Decode via ffmpeg to raw 12-bit YUVA (4 planes: Y, Cb, Cr, A).
    let decoded_path = tmp.join("decoded_a.yuv");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            patched_path.to_str().unwrap(),
            "-pix_fmt",
            "yuva444p12le",
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
        "ffmpeg failed to decode our interlaced 4444+alpha packet \
         (profile={profile_flag}, im={interlace_mode})"
    );
    let decoded = std::fs::read(&decoded_path).expect("read decoded");
    let plane_bytes = (width as usize) * (height as usize) * 2;
    assert!(
        decoded.len() >= plane_bytes * 4,
        "ffmpeg produced {} bytes, expected at least {} for 4 12-bit planes",
        decoded.len(),
        plane_bytes * 4
    );
    let decoded_y = &decoded[..plane_bytes];
    let decoded_a = &decoded[plane_bytes * 3..plane_bytes * 4];

    // Luma PSNR against the 12-bit source.
    let psnr = psnr_12bit(&src.planes[0].data, decoded_y);
    // Alpha is lossless per §7.1.2, but ffmpeg's ap4h/ap4x is internally
    // 12-bit so our 16-bit source alpha is resampled on decode. Compare
    // against the 12-bit-rounded source alpha (`round(s16 * 4095 /
    // 65535)`) and require the mean absolute error to stay sub-LSB.
    let n = plane_bytes / 2;
    let mut a_abs_err = 0u64;
    let mut a_min = u16::MAX;
    let mut a_max = 0u16;
    for i in 0..n {
        let s16 = u16::from_le_bytes([src.planes[3].data[i * 2], src.planes[3].data[i * 2 + 1]]);
        let s12 = ((s16 as u32 * 4095 + 32767) / 65535) as u16;
        let d = u16::from_le_bytes([decoded_a[i * 2], decoded_a[i * 2 + 1]]);
        a_min = a_min.min(d);
        a_max = a_max.max(d);
        a_abs_err += (d as i64 - s12 as i64).unsigned_abs();
    }
    let a_mae = a_abs_err as f64 / n as f64;
    eprintln!(
        "cross-decode interlaced 4444+alpha (profile={profile_flag}, im={interlace_mode}, \
         {width}x{height}, 12-bit): packet={} bytes, luma PSNR={psnr:.2} dB, \
         alpha range={a_min}..{a_max}, alpha MAE={a_mae:.4}",
        pkt.len()
    );
    assert!(
        psnr >= 30.0,
        "4444+alpha interlaced cross-decode PSNR {psnr:.2} dB under 30 dB bar"
    );
    // The alpha gradient must come through with a non-trivial range — a
    // dropped / mis-offset alpha blob collapses it to a constant.
    assert!(
        a_max - a_min > 2048,
        "decoded alpha range {a_min}..{a_max} is trivial — alpha blob lost?"
    );
    // Lossless-modulo-resample: mean abs error must be sub-LSB. A broken
    // alpha entropy decode would be tens-to-hundreds of levels off.
    assert!(
        a_mae < 1.0,
        "decoded alpha mean-abs-error {a_mae:.4} too high — alpha entropy decode wrong?"
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
        "4444+alpha interlaced cross-decode: even-row sum {even_sum} not > odd-row sum \
         {odd_sum} (TFF/BFF field assignment swapped in our encoder?)"
    );
}

#[test]
fn cross_decode_ap4h_interlaced_alpha_tff() {
    cross_decode_interlaced_4444_alpha(Profile::Prores4444, 64, 48, 1);
}

#[test]
fn cross_decode_ap4h_interlaced_alpha_bff() {
    cross_decode_interlaced_4444_alpha(Profile::Prores4444, 64, 48, 2);
}

#[test]
fn cross_decode_ap4h_interlaced_alpha_larger() {
    // 128x96 — twice the macroblock grid in the 4:4:4 + alpha field path.
    cross_decode_interlaced_4444_alpha(Profile::Prores4444, 128, 96, 1);
}

#[test]
fn cross_decode_ap4x_interlaced_alpha_tff() {
    // 4444 XQ shares the bitstream structure with 4444; this confirms the
    // field-pair + alpha path against the highest-quality profile too.
    cross_decode_interlaced_4444_alpha(Profile::Prores4444Xq, 64, 48, 1);
}

// ─────────── progressive 4444 + alpha (ap4h / ap4x single picture) ───────────
//
// The interlaced 4444+alpha cases above drive the §7.5.3 two-field
// deinterleave; this section covers the *progressive* counterpart — a
// single picture (interlace_mode = 0, `picture_count() == 1`) using the
// §7.2 Figure 4 progressive block scan instead of Figure 5. It is the
// symmetric forward (our-encoder → ffmpeg-decoder) acceptance for the
// crate's flagship progressive path, exercising at once:
//   * 4:4:4 chroma (ChromaFormat::Y444 — full-resolution Cb/Cr blocks),
//   * genuine 12-bit Y'CbCr samples (RDD 36 §7.5.1 level shift for
//     b = 12, `read_sample`'s `BitDepth::Twelve` branch) — ffmpeg's
//     ap4h/ap4x is internally 12-bit, so this matches its native depth,
//   * a per-pixel **16-bit alpha** plane coded losslessly per RDD 36
//     §5.3.3 + §7.1.2 (raster-scan run/diff VLC, Table 14) at the tail
//     of every slice, emitted for the full padded MB-row height (§7.5.2),
//   * the §7.2 Figure 4 *progressive* DCT-block scan (one picture, not a
//     field pair) — the path `encode_frame_with_alpha` takes when
//     `interlace_mode == 0`.
//
// ffmpeg's `prores_ks` decoder must reconstruct the single picture to
// ≥ 30 dB luma PSNR against the 12-bit source AND recover the alpha
// gradient. A regression in the progressive block scan or a dropped
// alpha blob both surface as failures.

/// Build a synthetic **progressive** 4:4:4 12-bit frame with a per-pixel
/// 16-bit alpha gradient. Unlike the interlaced generator there is no
/// even/odd field-brightness bias (a single picture has no field
/// structure); instead a smooth 2-D luma/chroma gradient + per-pixel
/// phase keeps AC coefficients non-zero across most blocks, and the
/// alpha sweeps the full 16-bit range diagonally so the alpha entropy
/// coder's run + difference codewords (Table 14) are exercised. Y/Cb/Cr
/// strides are in **bytes** (2 per 12-bit sample); alpha is 16-bit LE.
fn synthetic_progressive_444_alpha_12(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; w * h * 2];
    let mut cr = vec![0u8; w * h * 2];
    let mut a = vec![0u8; w * h * 2];
    let put = |buf: &mut [u8], idx: usize, v: u16| {
        let off = idx * 2;
        buf[off] = (v & 0xFF) as u8;
        buf[off + 1] = (v >> 8) as u8;
    };
    for j in 0..h {
        for i in 0..w {
            // Smooth 2-D luma ramp (left-dark → right-bright) inside a
            // 12-bit SMPTE-legal-ish window (≈256..3760), plus a per-pixel
            // phase so AC coefficients are non-zero across most blocks.
            let ramp = (i * 3000 / w.max(1)) as i32;
            let phase = ((i.wrapping_mul(7) ^ j.wrapping_mul(13)) % 384) as i32;
            let v = (400 + ramp + phase - 192).clamp(256, 3760) as u16;
            put(&mut y, j * w + i, v);
            // Full-resolution 4:4:4 chroma modulation.
            let cbv = (2048
                + ((i as i32 - w as i32 / 2) * 16 + (j as i32 % 7) * 8).clamp(-768, 768))
            .clamp(256, 3840) as u16;
            let crv = (2048
                + ((j as i32 - h as i32 / 2) * 16 + (i as i32 % 5) * 8).clamp(-768, 768))
            .clamp(256, 3840) as u16;
            put(&mut cb, j * w + i, cbv);
            put(&mut cr, j * w + i, crv);
            // 16-bit alpha diagonal gradient: 0 at (0,0) → ~65535 at the
            // far corner. `.max(1)` avoids a degenerate all-zero corner.
            let av = (((i + j) * 65535 / (w + h)) as u16).max(1);
            put(&mut a, j * w + i, av);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane {
                stride: w * 2,
                data: cb,
            },
            VideoPlane {
                stride: w * 2,
                data: cr,
            },
            VideoPlane {
                stride: w * 2,
                data: a,
            },
        ],
    }
}

/// Ask ffmpeg to produce a 1-frame **progressive** ProRes 4444/4444 XQ
/// MOV **with an alpha plane** (`yuva444p12le`). Used as a template
/// scaffold whose `mdat` payload we overwrite with our own encoder
/// output. No `+ildct` / `-top` flags — a plain progressive container so
/// the substituted packet lands in a progressive, alpha-aware scaffold.
fn ffmpeg_make_template_mov_444_alpha_progressive(
    profile_flag: u8,
    width: u32,
    height: u32,
    out_path: &std::path::Path,
) -> bool {
    let input = format!("testsrc=size={width}x{height}:rate=25:duration=1,format=yuva444p12le");
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
            "yuva444p12le",
            "-frames:v",
            "1",
            out_path.to_str().unwrap_or(""),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn cross_decode_progressive_4444_alpha(profile: Profile, width: u32, height: u32) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg missing — skipping 4444+alpha progressive cross-decode");
        return;
    }
    assert!(
        matches!(profile, Profile::Prores4444 | Profile::Prores4444Xq),
        "this driver is for the 4444 / 4444 XQ profiles only"
    );
    let tmp = tempdir().expect("tempdir");
    let profile_flag: u8 = match profile {
        Profile::Prores4444 => 4,
        Profile::Prores4444Xq => 5,
        _ => unreachable!("non-4444 profile in 4444 driver"),
    };
    let template_path = tmp.join(format!("template_ap_p{profile_flag}_{width}x{height}.mov"));
    if !ffmpeg_make_template_mov_444_alpha_progressive(profile_flag, width, height, &template_path)
    {
        eprintln!(
            "progressive 4444+alpha template MOV unavailable — skipping (profile={profile_flag})"
        );
        return;
    }
    let template = std::fs::read(&template_path).expect("read template");

    // Encode the same dimensions via our PROGRESSIVE 4444+alpha encoder at
    // genuine 12-bit 4:4:4 with a 16-bit alpha plane. This drives:
    //   * `read_sample`'s `BitDepth::Twelve` branch (§7.5.1 level shift),
    //   * the §7.1.2 / Table 14 16-bit-alpha entropy coder (per-slice
    //     scanned-alpha blob at the padded MB-row height, §7.5.2),
    //   * the §7.2 Figure 4 *progressive* block scan (single picture).
    let src = synthetic_progressive_444_alpha_12(width, height);
    let pkt = encode_frame_with_alpha(
        &src,
        width,
        height,
        ChromaFormat::Y444,
        BitDepth::Twelve,
        profile,
        2, // qi = 2 → finest at 4444 defaults
        Some(AlphaChannelType::Sixteen),
    )
    .expect("encode_frame_with_alpha 4444+alpha progressive");

    // Sanity: the encoded frame header reports progressive (interlace_mode
    // 0 ⇒ 1 picture) AND a non-zero alpha_channel_type.
    let (fh, _) = oxideav_prores::frame::parse_frame(&pkt).expect("parse our packet");
    assert_eq!(fh.interlace_mode, 0, "progressive frame interlace_mode");
    assert_eq!(
        fh.picture_count(),
        1,
        "progressive frame must carry exactly 1 picture"
    );
    assert_eq!(
        fh.alpha_channel_type, 2,
        "4444+alpha frame must report alpha_channel_type=2 (16-bit)"
    );

    let patched = patch_mov_with_packet(&template, &pkt);
    let patched_path = tmp.join(format!("patched_ap_p{profile_flag}_{width}x{height}.mov"));
    std::fs::write(&patched_path, &patched).expect("write patched");

    // Decode via ffmpeg to raw 12-bit YUVA (4 planes: Y, Cb, Cr, A).
    let decoded_path = tmp.join("decoded_ap.yuv");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            patched_path.to_str().unwrap(),
            "-pix_fmt",
            "yuva444p12le",
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
        "ffmpeg failed to decode our progressive 4444+alpha packet (profile={profile_flag})"
    );
    let decoded = std::fs::read(&decoded_path).expect("read decoded");
    let plane_bytes = (width as usize) * (height as usize) * 2;
    assert!(
        decoded.len() >= plane_bytes * 4,
        "ffmpeg produced {} bytes, expected at least {} for 4 12-bit planes",
        decoded.len(),
        plane_bytes * 4
    );
    let decoded_y = &decoded[..plane_bytes];
    let decoded_a = &decoded[plane_bytes * 3..plane_bytes * 4];

    // Luma PSNR against the 12-bit source.
    let psnr = psnr_12bit(&src.planes[0].data, decoded_y);
    // Alpha is lossless per §7.1.2, but ffmpeg's ap4h/ap4x is internally
    // 12-bit so our 16-bit source alpha is resampled on decode. Compare
    // against the 12-bit-rounded source alpha (`round(s16 * 4095 /
    // 65535)`) and require the mean absolute error to stay sub-LSB.
    let n = plane_bytes / 2;
    let mut a_abs_err = 0u64;
    let mut a_min = u16::MAX;
    let mut a_max = 0u16;
    for i in 0..n {
        let s16 = u16::from_le_bytes([src.planes[3].data[i * 2], src.planes[3].data[i * 2 + 1]]);
        let s12 = ((s16 as u32 * 4095 + 32767) / 65535) as u16;
        let d = u16::from_le_bytes([decoded_a[i * 2], decoded_a[i * 2 + 1]]);
        a_min = a_min.min(d);
        a_max = a_max.max(d);
        a_abs_err += (d as i64 - s12 as i64).unsigned_abs();
    }
    let a_mae = a_abs_err as f64 / n as f64;
    eprintln!(
        "cross-decode progressive 4444+alpha (profile={profile_flag}, {width}x{height}, \
         12-bit): packet={} bytes, luma PSNR={psnr:.2} dB, alpha range={a_min}..{a_max}, \
         alpha MAE={a_mae:.4}",
        pkt.len()
    );
    assert!(
        psnr >= 30.0,
        "4444+alpha progressive cross-decode PSNR {psnr:.2} dB under 30 dB bar"
    );
    // The alpha gradient must come through with a non-trivial range — a
    // dropped / mis-offset alpha blob collapses it to a constant.
    assert!(
        a_max - a_min > 2048,
        "decoded alpha range {a_min}..{a_max} is trivial — alpha blob lost?"
    );
    // Lossless-modulo-resample: mean abs error must be sub-LSB. A broken
    // alpha entropy decode would be tens-to-hundreds of levels off.
    assert!(
        a_mae < 1.0,
        "decoded alpha mean-abs-error {a_mae:.4} too high — alpha entropy decode wrong?"
    );

    // Left vs right luma sum check — the progressive source ramps
    // left-dark → right-bright, so a transposed / mis-scanned picture
    // (block-scan regression) shows up as a collapsed left/right bias.
    let mut left_sum = 0u64;
    let mut right_sum = 0u64;
    let w = width as usize;
    let half = w / 2;
    for j in 0..(height as usize) {
        let row_start = j * w * 2;
        for i in 0..w {
            let v = u16::from_le_bytes([
                decoded_y[row_start + i * 2],
                decoded_y[row_start + i * 2 + 1],
            ]) as u64;
            if i < half {
                left_sum += v;
            } else {
                right_sum += v;
            }
        }
    }
    assert!(
        right_sum > left_sum,
        "progressive 4444+alpha cross-decode: right-half sum {right_sum} not > left-half sum \
         {left_sum} (luma ramp lost — block-scan regression in our encoder?)"
    );
}

#[test]
fn cross_decode_ap4h_progressive_alpha() {
    cross_decode_progressive_4444_alpha(Profile::Prores4444, 64, 48);
}

#[test]
fn cross_decode_ap4h_progressive_alpha_larger() {
    // 128x96 — twice the macroblock grid in the 4:4:4 + alpha progressive path.
    cross_decode_progressive_4444_alpha(Profile::Prores4444, 128, 96);
}

#[test]
fn cross_decode_ap4x_progressive_alpha() {
    // 4444 XQ shares the bitstream structure with 4444; confirms the
    // progressive 4:4:4 + alpha path against the highest-quality profile too.
    cross_decode_progressive_4444_alpha(Profile::Prores4444Xq, 64, 48);
}
