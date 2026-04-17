//! ProRes encoder (422 Proxy / LT / Standard + 4444).
//!
//! Reads a `Yuv422P` or `Yuv444P` `VideoFrame`, walks 16x16 macroblocks,
//! forward DCTs each 8x8 block, quantises by `qmat * quant_index`, and
//! emits the frame/picture/slice container defined in `frame.rs`.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::dct::fdct8x8;
use crate::frame::{
    write_frame_header, write_picture_header, ChromaFormat, Profile, FRAME_HDR_SIZE,
    PICTURE_HDR_SIZE,
};
use crate::quant::{CHROMA_QMAT_4444, CHROMA_QMAT_STANDARD, LUMA_QMAT_4444, LUMA_QMAT_STANDARD};
use crate::slice::{blocks_per_mb, encode_slice, MAX_MBS_PER_SLICE};

/// Default `quant_index` used for 422 Standard. Lower = higher quality.
pub const DEFAULT_QUANT_INDEX: u8 = 4;

const MB_WIDTH_PX: usize = 16;
const MB_HEIGHT_PX: usize = 16;
const SLICE_MB_WIDTH_LOG2: u8 = 3; // 2^3 = 8 MBs per slice

/// Pick a profile from `bit_rate` when the caller expresses a target
/// rate, else fall back to Standard / 4444 depending on chroma.
///
/// The mapping is a rough proportional match to Apple's published
/// 1080p29.97 targets (Proxy 45, LT 102, Std 147, HQ 220, 4444 330,
/// 4444 XQ 500 Mbps) scaled to a picked breakpoint.
fn pick_profile(chroma: ChromaFormat, bit_rate: Option<u64>) -> Profile {
    match (chroma, bit_rate) {
        (ChromaFormat::Y422, Some(br)) if br <= 70_000_000 => Profile::Proxy,
        (ChromaFormat::Y422, Some(br)) if br <= 125_000_000 => Profile::Lt,
        (ChromaFormat::Y422, Some(br)) if br <= 180_000_000 => Profile::Standard,
        (ChromaFormat::Y422, Some(_)) => Profile::Hq,
        (ChromaFormat::Y422, None) => Profile::Standard,
        (ChromaFormat::Y444, Some(br)) if br >= 400_000_000 => Profile::Prores4444Xq,
        (ChromaFormat::Y444, _) => Profile::Prores4444,
    }
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("prores encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("prores encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv422P);

    let chroma = match pix {
        PixelFormat::Yuv422P => ChromaFormat::Y422,
        PixelFormat::Yuv444P => ChromaFormat::Y444,
        other => {
            return Err(Error::unsupported(format!(
                "prores encoder: pixel format {other:?} not supported (only Yuv422P / Yuv444P)"
            )));
        }
    };
    let profile = pick_profile(chroma, params.bit_rate);

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    let quant_index = profile.default_quant_index();

    Ok(Box::new(ProResEncoder {
        output_params,
        width,
        height,
        chroma,
        profile,
        quant_index,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct ProResEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    profile: Profile,
    quant_index: u8,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for ProResEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                if v.width != self.width || v.height != self.height {
                    return Err(Error::invalid(
                        "prores encoder: frame dimensions do not match encoder config",
                    ));
                }
                let expected_pix = match self.chroma {
                    ChromaFormat::Y422 => PixelFormat::Yuv422P,
                    ChromaFormat::Y444 => PixelFormat::Yuv444P,
                };
                if v.format != expected_pix {
                    return Err(Error::invalid(format!(
                        "prores encoder: frame format {:?} does not match configured {:?}",
                        v.format, expected_pix
                    )));
                }
                let data = encode_frame(v, self.chroma, self.profile, self.quant_index)?;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
                Ok(())
            }
            _ => Err(Error::invalid("prores encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Back-compat wrapper that encodes a 4:2:2 frame with the same API
/// shape as the pre-4444 implementation.
pub fn encode_frame_422(frame: &VideoFrame, profile: Profile, quant_index: u8) -> Result<Vec<u8>> {
    if frame.format != PixelFormat::Yuv422P {
        return Err(Error::unsupported("prores encoder: Yuv422P only"));
    }
    encode_frame(frame, ChromaFormat::Y422, profile, quant_index)
}

/// Encode a single picture (4:2:2 or 4:4:4) to a complete ProRes packet.
pub fn encode_frame(
    frame: &VideoFrame,
    chroma: ChromaFormat,
    profile: Profile,
    quant_index: u8,
) -> Result<Vec<u8>> {
    match chroma {
        ChromaFormat::Y422 => {
            if frame.format != PixelFormat::Yuv422P {
                return Err(Error::unsupported(
                    "prores encoder: Y422 chroma format requires Yuv422P",
                ));
            }
        }
        ChromaFormat::Y444 => {
            if frame.format != PixelFormat::Yuv444P {
                return Err(Error::unsupported(
                    "prores encoder: Y444 chroma format requires Yuv444P",
                ));
            }
        }
    }
    if frame.planes.len() != 3 {
        return Err(Error::invalid("prores encoder: expected 3 planes"));
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    let c_w = match chroma {
        ChromaFormat::Y422 => width.div_ceil(2),
        ChromaFormat::Y444 => width,
    };

    // Profile selects the Q matrix: Standard/Proxy/LT share the RDD 36
    // table; HQ and both 4444 tiers use the flatter all-4 matrix and
    // rely on `quant_index` for rate control.
    let (luma_qmat, chroma_qmat) = match profile {
        Profile::Hq | Profile::Prores4444 | Profile::Prores4444Xq => {
            (&LUMA_QMAT_4444, &CHROMA_QMAT_4444)
        }
        Profile::Proxy | Profile::Lt | Profile::Standard => {
            (&LUMA_QMAT_STANDARD, &CHROMA_QMAT_STANDARD)
        }
    };
    // Enforce the chroma/profile invariant.
    if profile.chroma_format() != chroma {
        return Err(Error::invalid(
            "prores encoder: profile chroma_format does not match requested chroma",
        ));
    }

    let mbs_x = width.div_ceil(MB_WIDTH_PX);
    let mbs_y = height.div_ceil(MB_HEIGHT_PX);
    let slice_mb_width = 1usize << SLICE_MB_WIDTH_LOG2;
    let slices_per_row = mbs_x.div_ceil(slice_mb_width);
    let slice_count = slices_per_row * mbs_y;
    let per_mb = blocks_per_mb(chroma);

    // 4 luma blocks per MB always live at these (bx,by) offsets (in 8-px units).
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

    // Chroma block layout inside a 16x16 MB:
    // * 4:2:2 — 2 blocks stacked vertically at x=0 (chroma is 8px wide per MB).
    // * 4:4:4 — 4 blocks in the same 2x2 layout as luma (chroma is full-res).
    let chroma_offsets: &[(usize, usize)] = match chroma {
        ChromaFormat::Y422 => &[(0, 0), (0, 1)],
        ChromaFormat::Y444 => &LUMA_OFFSETS,
    };

    // Build each slice payload.
    let mut slice_payloads: Vec<Vec<u8>> = Vec::with_capacity(slice_count);
    for my in 0..mbs_y {
        let mut mx = 0usize;
        while mx < mbs_x {
            let remaining = mbs_x - mx;
            let mbs_this_slice = remaining.min(MAX_MBS_PER_SLICE).min(slice_mb_width);
            let mut blocks: Vec<[i32; 64]> = Vec::with_capacity(mbs_this_slice * per_mb);
            for mb_within in 0..mbs_this_slice {
                let mb_x = mx + mb_within;
                // 4 luma blocks.
                for (bx, by) in LUMA_OFFSETS {
                    let x0 = mb_x * MB_WIDTH_PX + bx * 8;
                    let y0 = my * MB_HEIGHT_PX + by * 8;
                    blocks.push(encode_block(
                        &frame.planes[0].data,
                        frame.planes[0].stride,
                        width,
                        height,
                        x0,
                        y0,
                        luma_qmat,
                        quant_index,
                    ));
                }
                // Chroma blocks: Cb then Cr.
                for plane_idx in [1usize, 2] {
                    for (bx, by) in chroma_offsets.iter().copied() {
                        let (x0, y0) = match chroma {
                            ChromaFormat::Y422 => (mb_x * 8, my * MB_HEIGHT_PX + by * 8),
                            ChromaFormat::Y444 => {
                                (mb_x * MB_WIDTH_PX + bx * 8, my * MB_HEIGHT_PX + by * 8)
                            }
                        };
                        blocks.push(encode_block(
                            &frame.planes[plane_idx].data,
                            frame.planes[plane_idx].stride,
                            c_w,
                            height,
                            x0,
                            y0,
                            chroma_qmat,
                            quant_index,
                        ));
                    }
                }
            }
            let payload = encode_slice(mbs_this_slice as u8, quant_index, chroma, &blocks)?;
            slice_payloads.push(payload);
            mx += mbs_this_slice;
        }
    }
    debug_assert_eq!(slice_payloads.len(), slice_count);
    if slice_payloads.iter().any(|p| p.len() > u16::MAX as usize) {
        return Err(Error::invalid(
            "prores encoder: slice exceeded u16 size table limit",
        ));
    }

    // Compute sizes.
    let slice_table_size = slice_count * 2;
    let slice_bytes: usize = slice_payloads.iter().map(|p| p.len()).sum();
    let picture_size = (slice_table_size + slice_bytes) as u32;
    let total_frame_size =
        (FRAME_HDR_SIZE + PICTURE_HDR_SIZE + slice_table_size + slice_bytes) as u32;

    // Assemble.
    let mut out = Vec::with_capacity(total_frame_size as usize);
    write_frame_header(
        &mut out,
        frame.width as u16,
        frame.height as u16,
        chroma,
        profile,
        luma_qmat,
        chroma_qmat,
        total_frame_size,
    );
    write_picture_header(
        &mut out,
        picture_size,
        slice_count as u16,
        SLICE_MB_WIDTH_LOG2,
    );
    for p in &slice_payloads {
        out.extend_from_slice(&(p.len() as u16).to_be_bytes());
    }
    for p in &slice_payloads {
        out.extend_from_slice(p);
    }
    debug_assert_eq!(out.len(), total_frame_size as usize);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn encode_block(
    plane: &[u8],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    qmat: &[u8; 64],
    quant_index: u8,
) -> [i32; 64] {
    let mut blk = [0.0f32; 64];
    for j in 0..8 {
        let y = (y0 + j).min(plane_h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(plane_w.saturating_sub(1));
            blk[j * 8 + i] = plane[y * stride + x] as f32 - 128.0;
        }
    }
    fdct8x8(&mut blk);
    let qi = quant_index.max(1) as f32;
    let mut out = [0i32; 64];
    for k in 0..64 {
        let denom = qmat[k] as f32 * qi;
        let v = blk[k] / denom;
        out[k] = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -((-v + 0.5) as i32)
        };
    }
    out
}
