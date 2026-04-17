//! Slice pack / unpack for ProRes 422 and 4444.
//!
//! A slice covers a horizontal strip of up to 8 macroblocks (MBs) in
//! one MB row. For 4:2:2 each MB carries 4 luma blocks (arranged 2x2
//! horizontally, covering 16x16 pixels) and 2 Cb + 2 Cr chroma blocks
//! (each 8x8 — chroma is half-width, full-height). For 4:4:4 each MB
//! carries 4 luma + 4 Cb + 4 Cr blocks (chroma planes are now at full
//! luma resolution, so each chroma covers the full 16x16 MB area).
//!
//! Coded slice layout (one contiguous bitstream):
//!
//! ```text
//! 1 byte  : mb_count (1..=8)
//! 1 byte  : quant_index (scales both matrices)
//! entropy : for each MB: [DC se(v) | 63 AC se(v)] * blocks_per_mb
//! ```
//!
//! Where `blocks_per_mb` is 8 for 4:2:2 and 12 for 4:4:4 (luma-MB-order:
//! 4 Y, then the chroma blocks in the same 2x2 layout as luma for Cb
//! and Cr respectively).
//!
//! AC coefficients are walked in zig-zag order and emitted directly as
//! signed exp-Golomb values (no run/length pairing). This is
//! *larger* on the wire than RDD 36's run-level scheme, but byte-exact
//! and trivially self-consistent, which is what this minimal pair
//! needs. See `bitstream` module docs for the rationale.

use oxideav_core::{Error, Result};

use crate::bitstream::{BitReader, BitWriter};
use crate::frame::ChromaFormat;
use crate::quant::ZIGZAG;

/// 8 blocks per 4:2:2 macroblock (4Y + 2Cb + 2Cr).
pub const BLOCKS_PER_MB_422: usize = 8;

/// 12 blocks per 4:4:4 macroblock (4Y + 4Cb + 4Cr).
pub const BLOCKS_PER_MB_444: usize = 12;

/// Legacy alias kept for compatibility — refers to the 4:2:2 layout.
pub const BLOCKS_PER_MB: usize = BLOCKS_PER_MB_422;

/// Maximum MBs per slice. One slice never straddles an MB row.
pub const MAX_MBS_PER_SLICE: usize = 8;

/// Block count per macroblock for a given chroma format.
pub fn blocks_per_mb(chroma: ChromaFormat) -> usize {
    match chroma {
        ChromaFormat::Y422 => BLOCKS_PER_MB_422,
        ChromaFormat::Y444 => BLOCKS_PER_MB_444,
    }
}

/// A decoded slice: one 64-coeff block per plane per MB.
///
/// * 4:2:2 layout: `[MB0_Y0..Y3, MB0_Cb0..Cb1, MB0_Cr0..Cr1, MB1_...]`
/// * 4:4:4 layout: `[MB0_Y0..Y3, MB0_Cb0..Cb3, MB0_Cr0..Cr3, MB1_...]`
///
/// Coefficient storage is natural (row-major) order, post-dequant.
pub struct DecodedSlice {
    pub mb_count: u8,
    pub quant_index: u8,
    pub blocks: Vec<[i32; 64]>,
}

/// Encode a slice from a flat block list in the layout described in
/// [`DecodedSlice`]. `blocks.len()` must equal
/// `mb_count * blocks_per_mb(chroma)`.
/// Coefficients are **pre-quantised** ints in natural order — callers
/// run forward DCT + divide-by-quant-matrix before calling in.
pub fn encode_slice(
    mb_count: u8,
    quant_index: u8,
    chroma: ChromaFormat,
    blocks: &[[i32; 64]],
) -> Result<Vec<u8>> {
    if mb_count == 0 || mb_count as usize > MAX_MBS_PER_SLICE {
        return Err(Error::invalid("prores: slice mb_count out of range"));
    }
    let per_mb = blocks_per_mb(chroma);
    if blocks.len() != mb_count as usize * per_mb {
        return Err(Error::invalid("prores: slice block-count mismatch"));
    }
    let mut out = Vec::with_capacity(2 + 64 * blocks.len());
    out.push(mb_count);
    out.push(quant_index);

    let mut bw = BitWriter::new();
    for blk in blocks {
        // DC is the natural-order position 0.
        bw.write_se(blk[0]);
        // AC: 63 remaining coefficients in zig-zag order, directly signed-ue coded.
        for k in 1..64 {
            bw.write_se(blk[ZIGZAG[k] as usize]);
        }
    }
    out.extend(bw.finish());
    Ok(out)
}

/// Decode the inverse of [`encode_slice`]. Returns the quantised
/// coefficient blocks in natural order; caller applies inverse quant
/// and IDCT. The caller supplies `chroma` so the function knows how
/// many blocks per MB to consume.
pub fn decode_slice(data: &[u8], chroma: ChromaFormat) -> Result<DecodedSlice> {
    if data.len() < 2 {
        return Err(Error::invalid("prores: slice header truncated"));
    }
    let mb_count = data[0];
    let quant_index = data[1];
    if mb_count == 0 || mb_count as usize > MAX_MBS_PER_SLICE {
        return Err(Error::invalid("prores: slice mb_count out of range"));
    }
    let per_mb = blocks_per_mb(chroma);
    let expected_blocks = mb_count as usize * per_mb;
    let mut blocks = Vec::with_capacity(expected_blocks);

    let mut br = BitReader::new(&data[2..]);
    for _ in 0..expected_blocks {
        let mut blk = [0i32; 64];
        blk[0] = br.read_se()?;
        for k in 1..64 {
            blk[ZIGZAG[k] as usize] = br.read_se()?;
        }
        blocks.push(blk);
    }

    Ok(DecodedSlice {
        mb_count,
        quant_index,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_roundtrip() {
        let mut blocks = Vec::new();
        for mb in 0..3 {
            for blk_in_mb in 0..BLOCKS_PER_MB_422 {
                let mut blk = [0i32; 64];
                for k in 0..64 {
                    blk[k] = (((mb * 13 + blk_in_mb * 7 + k) as i32) % 31) - 15;
                }
                blocks.push(blk);
            }
        }
        let encoded = encode_slice(3, 4, ChromaFormat::Y422, &blocks).unwrap();
        let decoded = decode_slice(&encoded, ChromaFormat::Y422).unwrap();
        assert_eq!(decoded.mb_count, 3);
        assert_eq!(decoded.quant_index, 4);
        assert_eq!(decoded.blocks.len(), blocks.len());
        for (i, b) in decoded.blocks.iter().enumerate() {
            assert_eq!(*b, blocks[i], "block {i} differs");
        }
    }

    #[test]
    fn slice_roundtrip_444() {
        let mut blocks = Vec::new();
        for mb in 0..2 {
            for blk_in_mb in 0..BLOCKS_PER_MB_444 {
                let mut blk = [0i32; 64];
                for k in 0..64 {
                    blk[k] = (((mb * 17 + blk_in_mb * 5 + k) as i32) % 29) - 14;
                }
                blocks.push(blk);
            }
        }
        let encoded = encode_slice(2, 4, ChromaFormat::Y444, &blocks).unwrap();
        let decoded = decode_slice(&encoded, ChromaFormat::Y444).unwrap();
        assert_eq!(decoded.mb_count, 2);
        assert_eq!(decoded.blocks.len(), blocks.len());
        for (i, b) in decoded.blocks.iter().enumerate() {
            assert_eq!(*b, blocks[i], "444 block {i} differs");
        }
    }
}
