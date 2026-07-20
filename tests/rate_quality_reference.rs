//! Per-profile rate/quality acceptance against black-box reference
//! encodes of the *identical* source (RDD 36 §4 profiles).
//!
//! The cross-decode suites prove our streams decode externally at a
//! fixed PSNR floor; this suite measures **rate/quality per profile**
//! head-to-head:
//!
//! 1. A deterministic pseudo-natural 10-bit source is synthesised
//!    in-process and written as a raw planar file, so the reference
//!    encoder and our encoder consume byte-identical pixels (no
//!    `testsrc` re-render drift).
//! 2. The reference encoder produces a 1-frame MOV per profile from
//!    that raw file (black-box invocation only). Its `icpf` packet size
//!    is the *reference rate*; the MOV doubles as the container
//!    template for our packet.
//! 3. Our encoder encodes the same pixels twice: once at the profile's
//!    default `quantization_index` with the profile's signature
//!    quantisation matrices, and once **rate-controlled to the
//!    reference packet size** (equal-rate point).
//! 4. All three streams are decoded by the reference *decoder*
//!    (black-box) back to raw and scored as luma PSNR against the
//!    source.
//!
//! Acceptance per profile:
//! * the reference decoder accepts every stream we emit,
//! * our equal-rate packet lands within the rate-control tolerance
//!   envelope of the reference rate (±20 % guards the search's
//!   best-effort fallback),
//! * our equal-rate luma PSNR is within 6 dB of the reference
//!   encoder's PSNR at the same rate — an honest maturity bar for a
//!   spec-first encoder (no R-D optimisation, single qi per frame)
//!   against a production encoder with per-slice adaptive
//!   quantisation,
//! * our default-qi PSNR clears 40 dB on every profile (the default
//!   operating points are all visually-transparent presets).
//!
//! Skips gracefully when the reference binary is missing.

use std::process::Command;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::decoder::BitDepth;
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::{ChromaFormat, Profile};

const W: u32 = 256;
const H: u32 = 128;
const FPS: u64 = 25;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tempdir() -> Option<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = base.join(format!("oxideav-prores-ratequal-{pid}-{ts}-{seq}"));
    std::fs::create_dir_all(&p).ok()?;
    Some(p)
}

/// Deterministic pseudo-natural 10-bit source: layered low-frequency
/// gradients (compressible structure) + mid-frequency diagonal texture +
/// a small integer hash phase (keeps AC energy non-trivial in every
/// block, like sensor noise), inside a SMPTE-legal-ish window. Pure
/// integer math — identical bytes on every run and platform.
fn source_10bit(chroma: ChromaFormat) -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = match chroma {
        ChromaFormat::Y422 => w / 2,
        ChromaFormat::Y444 => w,
    };
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; cw * h * 2];
    let mut cr = vec![0u8; cw * h * 2];
    let put = |buf: &mut [u8], idx: usize, v: u16| {
        let off = idx * 2;
        buf[off] = (v & 0xFF) as u8;
        buf[off + 1] = (v >> 8) as u8;
    };
    let hash = |i: usize, j: usize| -> i32 {
        let x = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) >> 7;
        (x % 61) as i32 - 30
    };
    for j in 0..h {
        for i in 0..w {
            // Two crossed gradients + a triangle-wave texture.
            let g1 = (i * 1400 / w) as i32;
            let g2 = (j * 900 / h) as i32;
            let tri = {
                let t = (2 * i + 3 * j) % 256;
                if t < 128 {
                    t as i32
                } else {
                    255 - t as i32
                }
            };
            let v = (256 + g1 + g2 + tri * 3 + hash(i, j)).clamp(64, 940) as u16;
            put(&mut y, j * w + i, v);
        }
        for i in 0..cw {
            let si = match chroma {
                ChromaFormat::Y422 => i * 2,
                ChromaFormat::Y444 => i,
            };
            let cbv = (512 + ((si * 500 / w) as i32) - 250 + hash(si, j) / 2).clamp(64, 960) as u16;
            let crv = (512 + ((j * 400 / h) as i32) - 200 + hash(j, si) / 2).clamp(64, 960) as u16;
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

/// Serialise the source as a raw planar 10-bit LE file (Y then Cb then
/// Cr) — the layout the reference tools read via `-f rawvideo`.
fn write_raw(src: &VideoFrame, path: &std::path::Path) {
    let mut raw = Vec::new();
    for p in &src.planes {
        raw.extend_from_slice(&p.data);
    }
    std::fs::write(path, raw).expect("write raw source");
}

fn raw_pix_fmt(chroma: ChromaFormat) -> &'static str {
    match chroma {
        ChromaFormat::Y422 => "yuv422p10le",
        ChromaFormat::Y444 => "yuv444p10le",
    }
}

/// Reference-encode the raw source at `profile_flag`, returning the MOV
/// bytes (also used as the container template for our packets).
fn reference_encode(
    raw_path: &std::path::Path,
    chroma: ChromaFormat,
    profile_flag: u8,
    out_path: &std::path::Path,
) -> bool {
    Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            raw_pix_fmt(chroma),
            "-s",
            &format!("{W}x{H}"),
            "-r",
            &FPS.to_string(),
            "-i",
            raw_path.to_str().unwrap(),
            "-c:v",
            "prores_ks",
            "-profile:v",
            &profile_flag.to_string(),
            "-pix_fmt",
            raw_pix_fmt(chroma),
            "-frames:v",
            "1",
            out_path.to_str().unwrap(),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Black-box decode a MOV back to raw 10-bit planar.
fn reference_decode(
    mov_path: &std::path::Path,
    chroma: ChromaFormat,
    out_path: &std::path::Path,
) -> bool {
    Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            mov_path.to_str().unwrap(),
            "-pix_fmt",
            raw_pix_fmt(chroma),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            out_path.to_str().unwrap(),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Extract the first `icpf` ProRes packet from a MOV.
fn extract_prores_packet(mov: &[u8]) -> Option<Vec<u8>> {
    let needle = b"icpf";
    for i in 4..mov.len().checked_sub(4)? {
        if &mov[i..i + 4] == needle {
            let size_off = i - 4;
            let frame_size = u32::from_be_bytes(mov[size_off..size_off + 4].try_into().ok()?);
            let end = size_off + frame_size as usize;
            if end <= mov.len() {
                return Some(mov[size_off..end].to_vec());
            }
        }
    }
    None
}

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

/// Substitute our packet into the template MOV (same approach as
/// `tests/ffmpeg_cross_decode.rs`: replace the mdat payload, patch the
/// mdat size + the stsz sample size; stco stays valid because the mdat
/// payload offset is unchanged).
fn patch_mov_with_packet(template: &[u8], pkt: &[u8]) -> Vec<u8> {
    let (mdat_off, mdat_size) =
        find_top_atom(template, b"mdat", 0).expect("template MOV must contain mdat");
    let mdat_payload_start = mdat_off + 8;
    let mdat_payload_end = mdat_off + mdat_size as usize;
    let mut out =
        Vec::with_capacity(mdat_payload_start + pkt.len() + (template.len() - mdat_payload_end));
    out.extend_from_slice(&template[..mdat_payload_start]);
    out.extend_from_slice(pkt);
    out.extend_from_slice(&template[mdat_payload_end..]);
    let new_mdat_size = (8 + pkt.len()) as u32;
    out[mdat_off..mdat_off + 4].copy_from_slice(&new_mdat_size.to_be_bytes());
    let stsz_off = out
        .windows(4)
        .position(|w| w == b"stsz")
        .expect("stsz atom");
    let sample_size_off = stsz_off + 8;
    let sample_size = u32::from_be_bytes(
        out[sample_size_off..sample_size_off + 4]
            .try_into()
            .unwrap(),
    );
    if sample_size != 0 {
        out[sample_size_off..sample_size_off + 4]
            .copy_from_slice(&(pkt.len() as u32).to_be_bytes());
    } else {
        let first_entry_off = stsz_off + 16;
        out[first_entry_off..first_entry_off + 4]
            .copy_from_slice(&(pkt.len() as u32).to_be_bytes());
    }
    out
}

/// Luma PSNR between two 10-bit LE planes.
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

fn profile_flag(profile: Profile) -> u8 {
    match profile {
        Profile::Proxy => 0,
        Profile::Lt => 1,
        Profile::Standard => 2,
        Profile::Hq => 3,
        Profile::Prores4444 => 4,
        Profile::Prores4444Xq => 5,
        _ => unreachable!("variant not exercised by this test"),
    }
}

fn our_encode(
    src: &VideoFrame,
    chroma: ChromaFormat,
    profile: Profile,
    target_bytes: Option<usize>,
) -> Vec<u8> {
    let pix = match chroma {
        ChromaFormat::Y422 => PixelFormat::Yuv422P10Le,
        ChromaFormat::Y444 => PixelFormat::Yuv444P10Le,
    };
    let mut params = CodecParameters::video(CodecId::new("prores"));
    params.media_type = MediaType::Video;
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(pix);
    // Tag BT.709 colour metadata (§6.1.1 Tables 5/6): with the
    // all-zero "unknown" defaults the reference decoder guesses an RGB
    // (GBR) colourspace for the 4444 profiles and its format converter
    // refuses the reserved-primaries combination — the packet itself is
    // fine, but the comparison needs the decode to run.
    let mut cfg =
        EncoderConfig::signature_for_profile(profile).with_meta(oxideav_prores::frame::FrameMeta {
            aspect_ratio_information: 0,
            frame_rate_code: 0,
            color_primaries: 1,
            transfer_characteristic: 1,
            matrix_coefficients: 1,
        });
    if let Some(bytes) = target_bytes {
        params.bit_rate = Some(bytes as u64 * 8 * FPS);
        params.frame_rate = Some(oxideav_core::Rational::new(FPS as i64, 1));
        cfg = cfg.with_rate_control();
    }
    let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder_with_config");
    enc.send_frame(&Frame::Video(src.clone())).expect("send");
    enc.receive_packet().expect("receive").data
}

/// Run the full rate/quality comparison for one profile.
fn rate_quality(profile: Profile) {
    if !have_ffmpeg() {
        eprintln!("reference binary missing — skipping rate/quality test");
        return;
    }
    let chroma = profile.chroma_format();
    let flag = profile_flag(profile);
    let tmp = tempdir().expect("tempdir");
    let src = source_10bit(chroma);
    let raw_path = tmp.join("src.yuv");
    write_raw(&src, &raw_path);

    // 1. Reference encode (also the container template).
    let ref_mov_path = tmp.join(format!("ref_p{flag}.mov"));
    if !reference_encode(&raw_path, chroma, flag, &ref_mov_path) {
        eprintln!("reference encoder unavailable for profile {flag} — skipping");
        return;
    }
    let ref_mov = std::fs::read(&ref_mov_path).expect("read ref mov");
    let ref_pkt = extract_prores_packet(&ref_mov).expect("ref icpf packet");

    // 2. Our encodes: default-qi signature point + equal-rate point.
    let our_pkt_default = our_encode(&src, chroma, profile, None);
    let our_pkt_rate = our_encode(&src, chroma, profile, Some(ref_pkt.len()));

    // 3. Black-box decode all three; score luma PSNR against the source.
    let y_bytes = (W as usize) * (H as usize) * 2;
    let decode_and_score = |mov_bytes: &[u8], tag: &str| -> f64 {
        let mov_path = tmp.join(format!("{tag}_p{flag}.mov"));
        std::fs::write(&mov_path, mov_bytes).expect("write mov");
        let out_path = tmp.join(format!("{tag}_p{flag}.yuv"));
        assert!(
            reference_decode(&mov_path, chroma, &out_path),
            "reference decoder refused the {tag} stream (profile {flag})"
        );
        let decoded = std::fs::read(&out_path).expect("read decoded");
        assert!(decoded.len() >= y_bytes, "{tag}: short decode output");
        psnr_10bit(&src.planes[0].data, &decoded[..y_bytes])
    };
    let ref_psnr = decode_and_score(&ref_mov, "ref");
    let ours_default_mov = patch_mov_with_packet(&ref_mov, &our_pkt_default);
    let our_default_psnr = decode_and_score(&ours_default_mov, "ours-default");
    let ours_rate_mov = patch_mov_with_packet(&ref_mov, &our_pkt_rate);
    let our_rate_psnr = decode_and_score(&ours_rate_mov, "ours-eqrate");

    eprintln!(
        "rate/quality {profile:?} ({W}x{H} 10-bit): reference {} B @ {ref_psnr:.2} dB | \
         ours default-qi {} B @ {our_default_psnr:.2} dB | \
         ours equal-rate {} B @ {our_rate_psnr:.2} dB (delta {:+.2} dB)",
        ref_pkt.len(),
        our_pkt_default.len(),
        our_pkt_rate.len(),
        our_rate_psnr - ref_psnr,
    );

    // 4. Acceptance bars (see module docs).
    let rate_lo = ref_pkt.len() * 80 / 100;
    let rate_hi = ref_pkt.len() * 120 / 100;
    assert!(
        (rate_lo..=rate_hi).contains(&our_pkt_rate.len()),
        "{profile:?}: equal-rate packet {} B outside ±20 % of reference {} B",
        our_pkt_rate.len(),
        ref_pkt.len()
    );
    assert!(
        our_rate_psnr >= ref_psnr - 6.0,
        "{profile:?}: equal-rate PSNR {our_rate_psnr:.2} dB more than 6 dB under the \
         reference encoder's {ref_psnr:.2} dB at the same rate"
    );
    assert!(
        our_default_psnr >= 40.0,
        "{profile:?}: default-qi PSNR {our_default_psnr:.2} dB under the 40 dB \
         visually-transparent bar"
    );
}

#[test]
fn rate_quality_proxy() {
    rate_quality(Profile::Proxy);
}

#[test]
fn rate_quality_lt() {
    rate_quality(Profile::Lt);
}

#[test]
fn rate_quality_standard() {
    rate_quality(Profile::Standard);
}

#[test]
fn rate_quality_hq() {
    rate_quality(Profile::Hq);
}

#[test]
fn rate_quality_4444() {
    rate_quality(Profile::Prores4444);
}

#[test]
fn rate_quality_4444xq() {
    rate_quality(Profile::Prores4444Xq);
}

/// The self-roundtrip sanity leg: the equal-rate packet our encoder
/// produced for the HQ profile must decode through OUR decoder too (the
/// black-box leg above only proves the reference accepts it). Kept in
/// this suite so a rate-control change that emits a stream only the
/// reference tolerates (e.g. a slice-size accounting drift) fails here.
#[test]
fn equal_rate_packet_self_roundtrips() {
    if !have_ffmpeg() {
        return;
    }
    let chroma = ChromaFormat::Y422;
    let tmp = tempdir().expect("tempdir");
    let src = source_10bit(chroma);
    let raw_path = tmp.join("src.yuv");
    write_raw(&src, &raw_path);
    let ref_mov_path = tmp.join("ref_p3.mov");
    if !reference_encode(&raw_path, chroma, 3, &ref_mov_path) {
        return;
    }
    let ref_mov = std::fs::read(&ref_mov_path).expect("read ref mov");
    let ref_pkt = extract_prores_packet(&ref_mov).expect("ref icpf packet");
    let pkt = our_encode(&src, chroma, Profile::Hq, Some(ref_pkt.len()));
    let out = oxideav_prores::decoder::decode_packet_with_depth(
        &pkt,
        Some(0),
        Some((BitDepth::Ten, chroma)),
    )
    .expect("self-decode of the equal-rate packet");
    assert_eq!(out.planes.len(), 3);
    let psnr = psnr_10bit(&src.planes[0].data, &out.planes[0].data);
    eprintln!(
        "equal-rate HQ self-roundtrip: {} B @ {psnr:.2} dB",
        pkt.len()
    );
    assert!(psnr >= 40.0, "self-roundtrip PSNR {psnr:.2} dB under 40 dB");
}
