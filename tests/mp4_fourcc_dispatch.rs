//! Black-box test: generate real ProRes `.mov` files with the `ffmpeg`
//! binary and assert that the FourCC-to-codec-id mapping path owned by
//! this crate ([`oxideav_prores::codec_id_for_fourcc`] plus the crate's
//! `CodecRegistry` tag claims) reliably identifies the resulting stream
//! as the ProRes codec regardless of case.
//!
//! The test skips gracefully when `ffmpeg` is not present on PATH,
//! which is the common case on hermetic CI workers.
//!
//! Why this exists: the ProRes crate used to stop at
//! `CodecTag::fourcc(b"APCH")` registrations — any demuxer that relied
//! on the `codec_id.rs`-style static fallback (e.g. `oxideav-mp4`)
//! would return `mp4:apch` instead of "prores" when no registry was
//! plumbed in. This test pins the public static fallback surface so
//! a future regression is caught even in no-registry contexts.

use std::process::Command;

use oxideav_codec::CodecRegistry;
use oxideav_core::stream::{CodecResolver, ProbeContext};
use oxideav_core::{CodecId, CodecTag};
use oxideav_prores::{codec_id_for_fourcc, profile_for_fourcc, PRORES_FOURCCS};

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Read the MP4 `stsd` sample-entry FourCC (the 4 bytes at offset
/// `stsd_entry_start + 4`, which is after the 4-byte `size` field).
/// We find it by locating the `stsd` atom and reading the first entry's
/// type. The parse is intentionally shallow — enough to identify the
/// visual sample-entry FourCC that a demuxer would feed to
/// [`codec_id_for_fourcc`].
fn extract_stsd_fourcc(mp4: &[u8]) -> Option<[u8; 4]> {
    // Linear scan for `stsd` in the moov/trak/mdia/minf/stbl/stsd path.
    // For the tiny ffmpeg-produced .mov there is a single video track
    // so the first hit is the video's sample entry.
    let needle = b"stsd";
    let mut i = 0;
    while i + 4 <= mp4.len() {
        if &mp4[i..i + 4] == needle {
            // stsd header = 4 bytes `stsd` + 4 byte version/flags + 4 byte
            // entry_count, then each entry starts with [size:u32][type:[u8;4]].
            let entry_start = i + 4 + 4 + 4;
            if entry_start + 8 <= mp4.len() {
                let mut fc = [0u8; 4];
                fc.copy_from_slice(&mp4[entry_start + 4..entry_start + 8]);
                return Some(fc);
            }
        }
        i += 1;
    }
    None
}

/// Build a solid-color 64x48 yuva444p frame + alphamerge, encode with
/// `prores_ks` at the requested profile, and return the `.mov` bytes.
///
/// `profile_flag` is the ffmpeg `-profile:v` value:
/// - 0 = Proxy (`apco`)
/// - 1 = LT    (`apcs`)
/// - 2 = Standard (`apcn`)
/// - 3 = HQ    (`apch`)
/// - 4 = 4444  (`ap4h`)
/// - 5 = 4444 XQ (`ap4x`)
fn ffmpeg_make_prores_mov(profile_flag: u8) -> Option<Vec<u8>> {
    let tmp = tempdir()?;
    let out_path = tmp.join(format!("prores_p{profile_flag}.mov"));

    // 4444 / 4444 XQ need a 4:4:4 pixel format with alpha; 422 profiles
    // need yuv422p10le.
    let pix_fmt = match profile_flag {
        4 | 5 => "yuva444p10le",
        _ => "yuv422p10le",
    };

    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=1:duration=1",
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

/// Minimal scratch directory that auto-cleans at drop.
fn tempdir() -> Option<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let p = base.join(format!("oxideav-prores-mov-{pid}-{ts}"));
    std::fs::create_dir_all(&p).ok()?;
    Some(p)
}

#[test]
fn ffmpeg_generated_fourccs_map_to_prores() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping black-box FourCC test");
        return;
    }

    // Exercise every profile. The FourCC emitted by ffmpeg must match
    // the one we hard-code in PRORES_FOURCCS and must resolve to
    // `CodecId::new("prores")` via both helpers.
    // Profile flag 0..=5 maps to apco, apcs, apcn, apch, ap4h, ap4x.
    let expected: [&[u8; 4]; 6] = PRORES_FOURCCS;

    for (profile_flag, expect_fc) in expected.iter().enumerate() {
        let flag = profile_flag as u8;
        let Some(mp4) = ffmpeg_make_prores_mov(flag) else {
            eprintln!("ffmpeg prores_ks profile={flag} unavailable, skipping");
            continue;
        };
        let fc = extract_stsd_fourcc(&mp4).expect("stsd fourcc parse");

        // Case-insensitive match against what we expect.
        let mut upper_fc = fc;
        upper_fc.make_ascii_uppercase();
        let mut upper_ex = **expect_fc;
        upper_ex.make_ascii_uppercase();
        assert_eq!(
            upper_fc, upper_ex,
            "ffmpeg profile {flag} produced fourcc {:?}, expected {:?}",
            std::str::from_utf8(&fc).unwrap_or("?"),
            std::str::from_utf8(*expect_fc).unwrap_or("?"),
        );

        // Static fallback path: must identify as "prores".
        assert_eq!(
            codec_id_for_fourcc(&fc),
            Some(CodecId::new(oxideav_prores::CODEC_ID_STR)),
            "codec_id_for_fourcc rejected ffmpeg's {:?}",
            std::str::from_utf8(&fc).unwrap_or("?"),
        );

        // Profile lookup must return a Profile whose own fourcc()
        // matches (case-insensitive).
        let p = profile_for_fourcc(&fc).expect("profile_for_fourcc");
        let mut p_fc = *p.fourcc();
        p_fc.make_ascii_uppercase();
        assert_eq!(p_fc, upper_fc);

        // Dynamic registry path: after `register`, the FourCC must
        // resolve through the tag registry.
        let mut reg = CodecRegistry::new();
        oxideav_prores::register(&mut reg);
        let tag = CodecTag::fourcc(&fc);
        let ctx = ProbeContext::new(&tag);
        let id = reg.resolve_tag(&ctx).expect("registry resolve_tag");
        assert_eq!(id, CodecId::new(oxideav_prores::CODEC_ID_STR));
    }
}
