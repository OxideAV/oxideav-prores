//! Bit I/O primitives used by the SMPTE RDD 36 entropy coder.
//!
//! ProRes coefficients are coded with Golomb-Rice / exponential-Golomb
//! combination codes; the actual codeword decoders live in
//! [`crate::entropy`]. This module only provides the underlying MSB-first
//! bit reader / writer.

use oxideav_core::{Error, Result};

/// MSB-first bit reader with byte-aligned backing buffer.
pub struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize, // absolute bit offset from start of `buf`
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Read a single bit (0 or 1).
    pub fn read_bit(&mut self) -> Result<u32> {
        let byte_idx = self.bit_pos / 8;
        if byte_idx >= self.buf.len() {
            return Err(Error::invalid("prores bitstream: EOF"));
        }
        let bit_idx = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        Ok(((self.buf[byte_idx] >> bit_idx) & 1) as u32)
    }

    /// Read the next `n` bits (MSB first). `n` must be in `1..=32`.
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!((1..=32).contains(&n));
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()?;
        }
        Ok(v)
    }

    /// Approximation of the spec's `endOfData(dataSize)` predicate
    /// (RDD 36 §5.2): true if the number of remaining bits is 31 or
    /// fewer and any remaining bits are zero. This is used to terminate
    /// the AC-coefficient run-level loop.
    ///
    /// A correct implementation requires knowing the slice payload's
    /// exact byte length (the buffer we were constructed from is sized
    /// to that — `data` should be the per-component slice data only).
    pub fn end_of_data(&self) -> bool {
        let total_bits = self.buf.len() * 8;
        if self.bit_pos >= total_bits {
            return true;
        }
        let remaining = total_bits - self.bit_pos;
        if remaining > 31 {
            return false;
        }
        // Any remaining bit non-zero → not at EOD yet.
        let mut pos = self.bit_pos;
        while pos < total_bits {
            let byte_idx = pos / 8;
            let bit_idx = 7 - (pos % 8);
            if ((self.buf[byte_idx] >> bit_idx) & 1) != 0 {
                return false;
            }
            pos += 1;
        }
        true
    }

    /// How many whole bytes we've consumed, rounding up any partial byte.
    pub fn byte_pos(&self) -> usize {
        self.bit_pos.div_ceil(8)
    }

    /// Current bit position (absolute, from start of buffer).
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }
}

/// MSB-first bit writer.
///
/// Bits collect in a 64-bit accumulator and reach `buf` four bytes at a
/// time, so a codeword costs a shift and an OR rather than one
/// read-modify-write of the output per bit.
pub struct BitWriter {
    buf: Vec<u8>,
    /// Pending bits, right-aligned: the low `acc_bits` bits of `acc` are
    /// the next bits of the stream, oldest first. Bits above them are
    /// stale and never emitted.
    acc: u64,
    /// `0..32` between calls.
    acc_bits: u32,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// A writer whose output buffer starts with room for `bytes` bytes.
    pub fn with_capacity(bytes: usize) -> Self {
        Self::from_vec(Vec::with_capacity(bytes))
    }

    /// A writer that appends to `buf` after its existing bytes, reusing
    /// its allocation; [`finish`](Self::finish) hands it back.
    /// `byte_len` and `bit_len` include the existing bytes.
    pub fn from_vec(buf: Vec<u8>) -> Self {
        Self {
            buf,
            acc: 0,
            acc_bits: 0,
        }
    }

    pub fn write_bit(&mut self, b: u32) {
        self.write_bits(b & 1, 1);
    }

    /// Write the low `n` bits of `v`, most significant first.
    #[inline]
    pub fn write_bits(&mut self, v: u32, n: u32) {
        debug_assert!((1..=32).contains(&n));
        let v = v as u64 & ((1u64 << n) - 1);
        // acc_bits < 32 and n <= 32, so the pending bits fit in 64.
        self.acc = (self.acc << n) | v;
        self.acc_bits += n;
        if self.acc_bits >= 32 {
            self.acc_bits -= 32;
            let word = (self.acc >> self.acc_bits) as u32;
            self.buf.extend_from_slice(&word.to_be_bytes());
        }
    }

    /// Write `n` zero bits (any `n`, as in a unary prefix).
    pub fn write_zeros(&mut self, mut n: u32) {
        while n > 32 {
            self.write_bits(0, 32);
            n -= 32;
        }
        if n > 0 {
            self.write_bits(0, n);
        }
    }

    /// Pad to the next byte boundary with zero bits.
    pub fn align_byte(&mut self) {
        let pad = (8 - self.acc_bits % 8) % 8;
        if pad > 0 {
            self.write_bits(0, pad);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.align_byte();
        // Whole bytes remain: emit them oldest first.
        while self.acc_bits > 0 {
            self.acc_bits -= 8;
            self.buf.push((self.acc >> self.acc_bits) as u8);
        }
        self.buf
    }

    /// Bytes written so far, counting a partial trailing byte.
    pub fn byte_len(&self) -> usize {
        self.buf.len() + self.acc_bits.div_ceil(8) as usize
    }

    /// Number of bits written so far (including any partial trailing byte).
    pub fn bit_len(&self) -> usize {
        self.buf.len() * 8 + self.acc_bits as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010_1100, 8);
        w.write_bits(0x5A_5A, 16);
        w.write_bit(1);
        w.write_bit(0);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(8).unwrap(), 0b1010_1100);
        assert_eq!(r.read_bits(16).unwrap(), 0x5A_5A);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
    }

    #[test]
    fn end_of_data_with_zero_padding() {
        // Buffer of one byte 0b10000000 (just the bit '1' followed by 7 zero pads).
        let buf = [0b1000_0000u8];
        let mut r = BitReader::new(&buf);
        assert!(!r.end_of_data()); // 8 bits left, not at EOD
        let _ = r.read_bit().unwrap();
        // Now 7 zero bits remain; EOD should be true.
        assert!(r.end_of_data());
    }

    #[test]
    fn end_of_data_with_nonzero_padding() {
        let buf = [0b1000_0001u8];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bit().unwrap(); // consume the leading 1
        // Trailing bit is non-zero → not yet at EOD.
        assert!(!r.end_of_data());
    }

    #[test]
    fn writer_matches_bit_at_a_time_reference() {
        // Mixed-width writes straddling the 32-bit flush and byte
        // boundaries, checked against a one-bit-per-step reference.
        let mut seed = 0x1234_5678u32;
        let mut w = BitWriter::new();
        let mut bits = Vec::new();
        for i in 0..2000u32 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = 1 + (seed >> 27);
            if i % 7 == 0 {
                w.write_zeros(n + 33);
                bits.extend(std::iter::repeat_n(0u8, (n + 33) as usize));
            }
            w.write_bits(seed, n);
            bits.extend((0..n).rev().map(|b| ((seed >> b) & 1) as u8));
            assert_eq!(w.bit_len(), bits.len());
            assert_eq!(w.byte_len(), bits.len().div_ceil(8));
        }
        let mut want = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            want[i / 8] |= b << (7 - i % 8);
        }
        assert_eq!(w.finish(), want);
    }
}
