//! Profile target frame sizes and the encoder mode that holds frames to
//! them (`EncoderConfig::with_target_data_rate`).
//!
//! The targets are the "Target Data Rates" table of the Apple ProRes
//! white paper (April 2022): a target number of bits per frame for each
//! profile and frame size, the data rate being that times the frame
//! rate.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::decoder::decode_packet;
use oxideav_prores::encoder::{EncoderConfig, make_encoder_with_config};
use oxideav_prores::frame::{Profile, parse_frame, parse_picture_header, parse_slice_header};

const PROFILES: [Profile; 6] = [
    Profile::Proxy,
    Profile::Lt,
    Profile::Standard,
    Profile::Hq,
    Profile::Prores4444,
    Profile::Prores4444Xq,
];

fn megabits(profile: Profile, w: u32, h: u32, fps: f64) -> f64 {
    profile.target_frame_bytes(w, h) as f64 * 8.0 * fps / 1e6
}

#[test]
fn listed_sizes_reproduce_the_table() {
    // (width, height, frame rate, Mb/s per profile) rows of the table,
    // including frame rates other than the one the constants were read
    // from: the target is per frame, so every row must follow.
    let rows: [(u32, u32, f64, [f64; 6]); 7] = [
        (
            1920,
            1080,
            30000.0 / 1001.0,
            [45.0, 102.0, 147.0, 220.0, 330.0, 495.0],
        ),
        (1920, 1080, 24.0, [36.0, 82.0, 117.0, 176.0, 264.0, 396.0]),
        (1920, 1080, 25.0, [38.0, 85.0, 122.0, 184.0, 275.0, 413.0]),
        (
            720,
            486,
            30000.0 / 1001.0,
            [12.0, 29.0, 42.0, 63.0, 94.0, 141.0],
        ),
        (1280, 720, 25.0, [19.0, 42.0, 61.0, 92.0, 138.0, 206.0]),
        (
            3840,
            2160,
            30000.0 / 1001.0,
            [182.0, 410.0, 589.0, 884.0, 1326.0, 1989.0],
        ),
        (
            8192,
            4320,
            24.0,
            [622.0, 1400.0, 2011.0, 3017.0, 4525.0, 6788.0],
        ),
    ];
    for (w, h, fps, expect) in rows {
        for (profile, want) in PROFILES.into_iter().zip(expect) {
            let got = megabits(profile, w, h, fps);
            // The table prints whole Mb/s; 24p rows are derived from the
            // 23.976 fps rate. Allow the rounding of both.
            assert!(
                (got - want).abs() <= 0.5 + want * 0.012,
                "{profile:?} {w}x{h} @ {fps:.3}: {got:.1} Mb/s, table says {want}"
            );
        }
    }
}

#[test]
fn unlisted_sizes_scale_with_pixel_count() {
    for profile in PROFILES {
        // Half of 1920x1080 in each direction is nearest 720x576 in
        // pixel count and takes its bits per pixel.
        let quarter = profile.target_frame_bytes(960, 540);
        let pal = profile.target_frame_bytes(720, 576);
        let scaled = pal as u64 * (960 * 540) / (720 * 576);
        assert!((quarter as u64).abs_diff(scaled) <= 1, "{profile:?}");
        // Targets grow with the frame and with the profile.
        assert!(profile.target_frame_bytes(64, 64) < profile.target_frame_bytes(640, 480));
        assert!(profile.target_frame_bytes(1, 1) < 64);
    }
    for pair in PROFILES.windows(2) {
        assert!(pair[0].target_frame_bytes(1920, 1080) < pair[1].target_frame_bytes(1920, 1080));
    }
}

fn plane(w: usize, h: usize, seed: u32, noise: bool) -> VideoPlane {
    let mut s = seed;
    let mut data = Vec::with_capacity(w * h * 2);
    for j in 0..h {
        for i in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = if noise {
                // Texture well beyond what any profile target allows at
                // its default quantiser, yet codable under the target
                // by a coarser one.
                384 + ((i * 3 + j * 5) as u32 % 128) + ((s >> 24) & 127)
            } else {
                400 + ((i + j) as u32 / 16) % 64
            };
            data.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
    VideoPlane {
        stride: w * 2,
        data,
    }
}

fn encode(profile: Profile, w: u32, h: u32, noise: bool, config: EncoderConfig) -> Vec<u8> {
    let c444 = matches!(profile, Profile::Prores4444 | Profile::Prores4444Xq);
    let cw = if c444 { w as usize } else { w as usize / 2 };
    let mut params = CodecParameters::video(CodecId::new("prores"));
    params.media_type = MediaType::Video;
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(if c444 {
        PixelFormat::Yuv444P10Le
    } else {
        PixelFormat::Yuv422P10Le
    });
    let mut enc = make_encoder_with_config(&params, config).expect("encoder");
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            plane(w as usize, h as usize, 1, noise),
            plane(cw, h as usize, 2, noise),
            plane(cw, h as usize, 3, noise),
        ],
    }))
    .expect("send");
    enc.receive_packet().expect("packet").data
}

/// The quantisation index of the first slice of a frame.
fn first_slice_qi(packet: &[u8]) -> u8 {
    let (fh, rest) = parse_frame(packet).expect("frame");
    let (ph, after) = parse_picture_header(rest).expect("picture");
    let _ = ph;
    let mbs_x = (fh.width as usize).div_ceil(16);
    let mbs_y = (fh.height as usize).div_ceil(16);
    let slices = oxideav_prores::frame::compute_slice_sizes(mbs_x, 3).len() * mbs_y;
    let (sh, _) = parse_slice_header(&after[slices * 2..], false).expect("slice");
    sh.quantization_index
}

#[test]
fn hard_frames_are_held_to_the_target() {
    let (w, h) = (320, 192);
    for profile in PROFILES {
        let target = profile.target_frame_bytes(w, h);
        let free = encode(profile, w, h, true, EncoderConfig::for_profile(profile));
        assert!(
            free.len() > target * 3 / 2,
            "{profile:?}: noise must overshoot the target for this test to mean anything"
        );
        for threads in [1, 3] {
            let capped = encode(
                profile,
                w,
                h,
                true,
                EncoderConfig::for_profile(profile)
                    .with_target_data_rate()
                    .with_threads(threads),
            );
            assert!(
                capped.len() <= target + target / 20,
                "{profile:?}: {} bytes exceeds the {target}-byte target by more than 5 %",
                capped.len()
            );
            // Not needlessly small either: one index finer no longer fits,
            // or the frame is within 10 % of the target.
            let qi = first_slice_qi(&capped);
            assert!(qi > profile.default_quant_index());
            if capped.len() < target - target / 10 {
                let finer = encode(
                    profile,
                    w,
                    h,
                    true,
                    EncoderConfig::for_profile(profile).with_quantization_index(qi - 1),
                );
                assert!(
                    finer.len() > target + target / 20,
                    "{profile:?}: index {} would also fit ({} bytes)",
                    qi - 1,
                    finer.len()
                );
            }
            decode_packet(&capped, None).expect("capped frame decodes");
        }
    }
}

#[test]
fn frames_under_the_target_are_untouched() {
    let (w, h) = (320, 192);
    for profile in PROFILES {
        let free = encode(profile, w, h, false, EncoderConfig::for_profile(profile));
        assert!(free.len() <= profile.target_frame_bytes(w, h));
        let capped = encode(
            profile,
            w,
            h,
            false,
            EncoderConfig::for_profile(profile).with_target_data_rate(),
        );
        assert!(
            free == capped,
            "{profile:?}: an easy frame was re-quantised"
        );
    }
}
