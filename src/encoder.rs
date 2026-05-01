//! ProRes encoder following SMPTE RDD 36 §5 + §7.
//!
//! Reads a `Yuv422P` or `Yuv444P` `VideoFrame`, walks 16x16 macroblocks,
//! forward DCTs each 8x8 block, quantises by `qmat * qScale / 8`,
//! per-component slice-scans, and emits the entropy-coded slice payload
//! per RDD 36.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::alpha::{encode_scanned_alpha, AlphaChannelType};
use crate::dct::fdct8x8;
use crate::decoder::BitDepth;
use crate::frame::{
    compute_slice_sizes, write_frame_with_alpha, write_picture_header, write_slice_header,
    ChromaFormat, Profile,
};
use crate::quant::{qscale, DEFAULT_QMAT};
use crate::slice::{blocks_per_mb, chroma_blocks_per_mb, encode_slice_components};

/// Default `quantization_index` used for 422 Standard. Lower = higher quality.
pub const DEFAULT_QUANT_INDEX: u8 = 4;

/// Internal cap on encoded packet size — bounds the output Vec against
/// `width * height * bytes-per-sample * a small constant`. Prevents a
/// pathological caller (e.g. a header that lies about dimensions) from
/// driving an unbounded allocation.
fn output_capacity_cap(width: u16, height: u16, chroma: ChromaFormat) -> usize {
    let pixels = width as usize * height as usize;
    // A worst-case ProRes packet for 8-bit YUV is ~10 bytes per pixel
    // before container overhead. Pad to 16 + a slack for headers.
    let bpp = match chroma {
        ChromaFormat::Y422 => 16,
        ChromaFormat::Y444 => 24,
    };
    pixels.saturating_mul(bpp).saturating_add(1 << 16)
}

const MB_SIDE_PX: usize = 16;
const SLICE_MB_WIDTH_LOG2: u8 = 3; // 8 MBs per slice (typical)

/// Pick a profile from `bit_rate` when the caller expresses a target rate.
///
/// Public so callers can preview the encoder's profile selection without
/// running an encode (the chosen profile is no longer carried in the
/// RDD 36 bitstream — it lives at the container level via FourCC).
pub fn pick_profile(chroma: ChromaFormat, bit_rate: Option<u64>) -> Profile {
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

    let (chroma, bit_depth) = match pix {
        PixelFormat::Yuv422P => (ChromaFormat::Y422, BitDepth::Eight),
        PixelFormat::Yuv444P => (ChromaFormat::Y444, BitDepth::Eight),
        PixelFormat::Yuv422P10Le => (ChromaFormat::Y422, BitDepth::Ten),
        PixelFormat::Yuv444P10Le => (ChromaFormat::Y444, BitDepth::Ten),
        PixelFormat::Yuv422P12Le => (ChromaFormat::Y422, BitDepth::Twelve),
        PixelFormat::Yuv444P12Le => (ChromaFormat::Y444, BitDepth::Twelve),
        other => {
            return Err(Error::unsupported(format!(
                "prores encoder: pixel format {other:?} not supported \
                 (expected Yuv4(2|4)4P / Yuv4(2|4)4P10Le / Yuv4(2|4)4P12Le)"
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
        bit_depth,
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
    bit_depth: BitDepth,
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
                let data = encode_frame_with_depth(
                    v,
                    self.width,
                    self.height,
                    self.chroma,
                    self.bit_depth,
                    self.profile,
                    self.quant_index,
                )?;
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
/// shape as the pre-RDD 36 implementation.
pub fn encode_frame_422(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    profile: Profile,
    quant_index: u8,
) -> Result<Vec<u8>> {
    encode_frame(
        frame,
        width,
        height,
        ChromaFormat::Y422,
        profile,
        quant_index,
    )
}

/// Encode a single picture (4:2:2 or 4:4:4) to a complete RDD 36 frame.
/// 8-bit input only; for 10-bit see [`encode_frame_with_depth`].
pub fn encode_frame(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    profile: Profile,
    quantization_index: u8,
) -> Result<Vec<u8>> {
    encode_frame_with_depth(
        frame,
        img_w,
        img_h,
        chroma,
        BitDepth::Eight,
        profile,
        quantization_index,
    )
}

/// Encode a single picture to an RDD 36 frame at the requested bit depth.
///
/// `BitDepth::Eight` reads each sample as one byte; the deeper-bit paths
/// read each sample as a little-endian `u16` whose value is bounded by
/// the depth (`[0, 1023]` for 10-bit, `[0, 4095]` for 12-bit; high bits
/// ignored). Internal DCT precision is the same for all depths — input
/// samples are level-shifted into the spec's centred range
/// `v = s / 2^(b-9) - 256` (RDD 36 §7.5.1) so the quant-matrix and
/// qScale tables apply identically across depths.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_with_depth(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
) -> Result<Vec<u8>> {
    encode_frame_with_alpha(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        None,
    )
}

/// Encode a single picture to an RDD 36 frame with optional alpha
/// channel coding (RDD 36 §5.3.3 + §7.1.2).
///
/// When `alpha_channel_type` is `Some`, the input frame must carry a
/// 4th `VideoPlane` with a per-pixel alpha array at full luma resolution.
/// Each sample is read as one byte (`Eight`) — alpha values are
/// promoted to the spec's 16-bit internal representation and emitted as
/// `scanned_alpha()` blobs at the tail of every slice. The frame header
/// `alpha_channel_type` field is set accordingly.
///
/// When `alpha_channel_type` is `None`, behaviour is identical to
/// [`encode_frame_with_depth`] (3-plane input, no alpha emission).
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_with_alpha(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    alpha_channel_type: Option<AlphaChannelType>,
) -> Result<Vec<u8>> {
    encode_frame_full(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        alpha_channel_type,
        0,
    )
}

/// Encode an interlaced RDD 36 frame. `interlace_mode` selects the
/// field order (1 = top-field-first, 2 = bottom-field-first). The
/// supplied frame's planes are sliced into top + bottom field pictures
/// per §7.5.3 (rows {0, 2, …} → top, rows {1, 3, …} → bottom) and each
/// field is encoded as a separate `picture()` per §5.1, sharing one
/// frame_header().
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_interlaced(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    alpha_channel_type: Option<AlphaChannelType>,
    interlace_mode: u8,
) -> Result<Vec<u8>> {
    if interlace_mode != 1 && interlace_mode != 2 {
        return Err(Error::invalid(
            "prores encoder: encode_frame_interlaced requires interlace_mode in {1, 2}",
        ));
    }
    encode_frame_full(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        alpha_channel_type,
        interlace_mode,
    )
}

/// Internal entrypoint shared by progressive and interlaced encodes.
/// `interlace_mode == 0` builds a single picture; `1` (TFF) or `2`
/// (BFF) builds two field pictures per §5.1.
#[allow(clippy::too_many_arguments)]
fn encode_frame_full(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    alpha_channel_type: Option<AlphaChannelType>,
    interlace_mode: u8,
) -> Result<Vec<u8>> {
    let expected_planes = if alpha_channel_type.is_some() { 4 } else { 3 };
    if frame.planes.len() != expected_planes {
        return Err(Error::invalid(format!(
            "prores encoder: expected {expected_planes} planes (got {})",
            frame.planes.len()
        )));
    }
    if !(1..=224).contains(&quantization_index) {
        return Err(Error::invalid(
            "prores encoder: quantization_index out of range",
        ));
    }
    if profile.chroma_format() != chroma {
        return Err(Error::invalid(
            "prores encoder: profile chroma_format does not match requested chroma",
        ));
    }
    let width = img_w as usize;
    let height = img_h as usize;

    // Bound output capacity against header-declared dimensions.
    let cap = output_capacity_cap(img_w as u16, img_h as u16, chroma);

    // Use the default flat all-4 quant matrix; do not load custom matrices.
    let qmat = &DEFAULT_QMAT;

    // Per §6.2 picture_vertical_size derivation. Each interlaced field
    // is a separate picture sized at half the frame height (rounded
    // appropriately for top vs. bottom).
    let pictures: Vec<(usize, FieldStride)> = if interlace_mode == 0 {
        vec![(height, FieldStride::progressive())]
    } else {
        let top_h = height.div_ceil(2);
        let bot_h = height / 2;
        // interlace_mode 1: first picture is top field (offset 0)
        // interlace_mode 2: first picture is bottom field (offset 1)
        if interlace_mode == 1 {
            vec![
                (top_h, FieldStride::new(2, 0)),
                (bot_h, FieldStride::new(2, 1)),
            ]
        } else {
            vec![
                (bot_h, FieldStride::new(2, 1)),
                (top_h, FieldStride::new(2, 0)),
            ]
        }
    };

    let interlaced = interlace_mode != 0;
    let mut picture_blobs: Vec<Vec<u8>> = Vec::with_capacity(pictures.len());
    for (picture_height, field) in &pictures {
        let blob = encode_one_picture(
            frame,
            width,
            height,
            *picture_height,
            chroma,
            bit_depth,
            quantization_index,
            qmat,
            alpha_channel_type,
            interlaced,
            *field,
        )?;
        picture_blobs.push(blob);
    }

    let frame_header_size = 20usize; // no qmats loaded → 20 bytes
    let pictures_total: usize = picture_blobs.iter().map(|p| p.len()).sum();
    let total_frame_size_no_padding = 4 + 4 + frame_header_size + pictures_total;
    if total_frame_size_no_padding > cap {
        return Err(Error::invalid(
            "prores encoder: encoded size exceeds internal cap",
        ));
    }

    let mut out = Vec::with_capacity(total_frame_size_no_padding);
    write_frame_with_alpha(
        &mut out,
        total_frame_size_no_padding as u32,
        img_w as u16,
        img_h as u16,
        chroma,
        interlace_mode,
        qmat,
        qmat,
        false, // load_luma_qmat = 0 → default
        false, // load_chroma_qmat = 0
        alpha_channel_type.map_or(0, |a| a.code()),
    );
    for blob in &picture_blobs {
        out.extend_from_slice(blob);
    }
    debug_assert_eq!(out.len(), total_frame_size_no_padding);
    Ok(out)
}

/// Field-row mapping for source-plane reads on the encoder side.
/// Mirrors `decoder::FieldStride`.
#[derive(Copy, Clone, Debug)]
struct FieldStride {
    step: usize,
    offset: usize,
}

impl FieldStride {
    fn new(step: usize, offset: usize) -> Self {
        Self { step, offset }
    }
    fn progressive() -> Self {
        Self { step: 1, offset: 0 }
    }
    fn map(self, picture_row: usize) -> usize {
        self.step * picture_row + self.offset
    }
}

/// Build one `picture()` blob (picture_header + slice_table +
/// concatenated slice payloads). For interlaced encodes the caller
/// invokes this twice (once per field).
#[allow(clippy::too_many_arguments)]
fn encode_one_picture(
    frame: &VideoFrame,
    frame_w: usize,
    frame_h: usize,
    picture_height: usize,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    quantization_index: u8,
    qmat: &[u8; 64],
    alpha_channel_type: Option<AlphaChannelType>,
    interlaced: bool,
    field: FieldStride,
) -> Result<Vec<u8>> {
    let c_w = match chroma {
        ChromaFormat::Y422 => frame_w.div_ceil(2),
        ChromaFormat::Y444 => frame_w,
    };
    let mbs_x = frame_w.div_ceil(MB_SIDE_PX);
    let mbs_y = picture_height.div_ceil(MB_SIDE_PX);
    let slice_sizes_template = compute_slice_sizes(mbs_x, SLICE_MB_WIDTH_LOG2);
    let slices_per_row = slice_sizes_template.len();
    let slice_count = slices_per_row * mbs_y;
    let _cb_per_mb = chroma_blocks_per_mb(chroma);
    let per_mb = blocks_per_mb(chroma);

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let chroma_offsets: &[(usize, usize)] = match chroma {
        ChromaFormat::Y422 => &[(0, 0), (0, 1)],
        ChromaFormat::Y444 => &LUMA_OFFSETS,
    };

    let mut slice_payloads: Vec<Vec<u8>> = Vec::with_capacity(slice_count);
    for my in 0..mbs_y {
        let mut mx = 0usize;
        for &mbs_this_slice in &slice_sizes_template {
            let mbs_this_slice = mbs_this_slice.min(mbs_x - mx);
            if mbs_this_slice == 0 {
                break;
            }
            let mut blocks: Vec<[i32; 64]> = Vec::with_capacity(mbs_this_slice * per_mb);
            for mb_within in 0..mbs_this_slice {
                let mb_x = mx + mb_within;
                for (bx, by) in LUMA_OFFSETS {
                    let x0 = mb_x * MB_SIDE_PX + bx * 8;
                    let y0 = my * MB_SIDE_PX + by * 8;
                    blocks.push(encode_block(
                        &frame.planes[0].data,
                        frame.planes[0].stride,
                        frame_w,
                        frame_h,
                        x0,
                        y0,
                        qmat,
                        quantization_index,
                        bit_depth,
                        field,
                    ));
                }
                for plane_idx in [1usize, 2] {
                    for (bx, by) in chroma_offsets.iter().copied() {
                        let (x0, y0) = match chroma {
                            ChromaFormat::Y422 => (mb_x * 8, my * MB_SIDE_PX + by * 8),
                            ChromaFormat::Y444 => {
                                (mb_x * MB_SIDE_PX + bx * 8, my * MB_SIDE_PX + by * 8)
                            }
                        };
                        blocks.push(encode_block(
                            &frame.planes[plane_idx].data,
                            frame.planes[plane_idx].stride,
                            c_w,
                            frame_h,
                            x0,
                            y0,
                            qmat,
                            quantization_index,
                            bit_depth,
                            field,
                        ));
                    }
                }
            }
            let (y_data, cb_data, cr_data) =
                encode_slice_components(mbs_this_slice, chroma, interlaced, &blocks)?;
            if y_data.len() > u16::MAX as usize
                || cb_data.len() > u16::MAX as usize
                || cr_data.len() > u16::MAX as usize
            {
                return Err(Error::invalid(
                    "prores encoder: slice component exceeded u16 size limit",
                ));
            }

            let alpha_blob: Vec<u8> = if let Some(act) = alpha_channel_type {
                let slice_vertical_size = if my < mbs_y - 1 {
                    MB_SIDE_PX
                } else {
                    picture_height - my * MB_SIDE_PX
                };
                let cols = MB_SIDE_PX * mbs_this_slice;
                let mut samples: Vec<u16> = Vec::with_capacity(cols * slice_vertical_size);
                let a_plane = &frame.planes[3];
                let a_stride = a_plane.stride;
                for r in 0..slice_vertical_size {
                    let frame_row = field
                        .map(my * MB_SIDE_PX + r)
                        .min(frame_h.saturating_sub(1));
                    for c in 0..cols {
                        let x = (mx * MB_SIDE_PX + c).min(frame_w.saturating_sub(1));
                        let v: u16 = match act {
                            AlphaChannelType::Eight => {
                                a_plane.data[frame_row * a_stride + x] as u16
                            }
                            AlphaChannelType::Sixteen => {
                                let off = frame_row * a_stride + x * 2;
                                u16::from_le_bytes([a_plane.data[off], a_plane.data[off + 1]])
                            }
                        };
                        samples.push(v);
                    }
                }
                encode_scanned_alpha(&samples, act)?
            } else {
                Vec::new()
            };

            let cr_field = if alpha_channel_type.is_some() {
                Some(cr_data.len() as u16)
            } else {
                None
            };
            let mut slice_buf = Vec::with_capacity(
                8 + y_data.len() + cb_data.len() + cr_data.len() + alpha_blob.len(),
            );
            write_slice_header(
                &mut slice_buf,
                quantization_index,
                y_data.len() as u16,
                cb_data.len() as u16,
                cr_field,
            );
            slice_buf.extend_from_slice(&y_data);
            slice_buf.extend_from_slice(&cb_data);
            slice_buf.extend_from_slice(&cr_data);
            slice_buf.extend_from_slice(&alpha_blob);
            slice_payloads.push(slice_buf);
            mx += mbs_this_slice;
        }
    }
    debug_assert_eq!(slice_payloads.len(), slice_count);
    if slice_payloads.iter().any(|p| p.len() > u16::MAX as usize) {
        return Err(Error::invalid(
            "prores encoder: slice exceeded u16 size table limit",
        ));
    }

    let slice_table_size = slice_count * 2;
    let slice_bytes: usize = slice_payloads.iter().map(|p| p.len()).sum();
    let picture_header_size = 8usize;
    let picture_size = (picture_header_size + slice_table_size + slice_bytes) as u32;
    let mut blob = Vec::with_capacity(picture_size as usize);
    write_picture_header(
        &mut blob,
        picture_size,
        if slice_count <= u16::MAX as usize {
            slice_count as u16
        } else {
            0
        },
        SLICE_MB_WIDTH_LOG2,
    );
    for p in &slice_payloads {
        blob.extend_from_slice(&(p.len() as u16).to_be_bytes());
    }
    for p in &slice_payloads {
        blob.extend_from_slice(p);
    }
    debug_assert_eq!(blob.len(), picture_size as usize);
    Ok(blob)
}

/// Sample one IDCT input value from the source plane at sample
/// coordinate `(x, y)`, applying the spec's level-shift to a centred
/// `v` in the range `[-256, 256)` per RDD 36 §7.5.1. The inverse of
/// the decoder formula `s = 2^b * (v + 256) / 512` is
/// `v = s * 512 / 2^b - 256 = s / 2^(b-9) - 256`. `stride` is in
/// **bytes**; for 10/12-bit planes that's `2 * samples_per_row`.
fn read_sample(plane: &[u8], stride: usize, x: usize, y: usize, bit_depth: BitDepth) -> f32 {
    match bit_depth {
        BitDepth::Eight => (plane[y * stride + x] as f32) * 2.0 - 256.0,
        BitDepth::Ten => {
            let off = y * stride + x * 2;
            let lo = plane[off] as u16;
            let hi = plane[off + 1] as u16;
            let s = (lo | (hi << 8)) & 0x03FF;
            (s as f32) / 2.0 - 256.0
        }
        BitDepth::Twelve => {
            let off = y * stride + x * 2;
            let lo = plane[off] as u16;
            let hi = plane[off + 1] as u16;
            let s = (lo | (hi << 8)) & 0x0FFF;
            (s as f32) / 8.0 - 256.0
        }
    }
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
    quantization_index: u8,
    bit_depth: BitDepth,
    field: FieldStride,
) -> [i32; 64] {
    let mut blk = [0.0f32; 64];
    for j in 0..8 {
        // Map per-picture row to per-frame row; this is the identity for
        // progressive (`step=1, offset=0`) and 2*r+offset for interlaced.
        let frame_row = field.map(y0 + j).min(plane_h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(plane_w.saturating_sub(1));
            blk[j * 8 + i] = read_sample(plane, stride, x, frame_row, bit_depth);
        }
    }
    fdct8x8(&mut blk);
    // Quantisation: F[v][u] = (QF[v][u] * W[v][u] * qScale) / 8
    // Inverse: QF = round(F * 8 / (W * qScale)).
    let qs = qscale(quantization_index) as f32;
    let mut out = [0i32; 64];
    for k in 0..64 {
        let denom = qmat[k] as f32 * qs;
        let v = blk[k] * 8.0 / denom;
        out[k] = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -((-v + 0.5) as i32)
        };
    }
    out
}
