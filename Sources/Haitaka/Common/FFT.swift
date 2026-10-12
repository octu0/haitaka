import Foundation

/// In-place radix-2 Cooley-Tukey FFT. `size` must be a power of two.
public struct FFT: Sendable {
    public let size: Int
    public let log2Size: Int
    
    private let bitReversedIndices: [Int]
    /// Twiddle factors laid out contiguously per stage so the butterfly loop can load them with SIMD.
    /// Factor j of stage s is at `stageTwiddleOffsets[s] + j` (instead of the strided `j * step`).
    private let stageTwiddleReal: [Float]
    private let stageTwiddleImag: [Float]
    private let stageTwiddleOffsets: [Int]
    
    public init(size: Int = 512) {
        self.size = size
        
        var m = 0
        var temp = size
        while 1 < temp {
            temp = temp >> 1
            m += 1
        }
        self.log2Size = m
        
        // Bit-reversal permutation table
        var bitRev = [Int](repeating: 0, count: size)
        var i = 0
        while i < size {
            var rev = 0
            var b = 0
            while b < m {
                if (i & (1 << b)) != 0 {
                    rev |= (1 << ((m - 1) - b))
                }
                b += 1
            }
            bitRev[i] = rev
            i += 1
        }
        self.bitReversedIndices = bitRev
        
        // Twiddle factors e^{-j 2 pi k / N}
        let half = size / 2
        var twReal = [Float](repeating: 0.0, count: half)
        var twImag = [Float](repeating: 0.0, count: half)
        
        var k = 0
        let factor = (2.0 * Float.pi) / Float(size)
        while k < half {
            let theta = -1.0 * Float(k) * factor
            twReal[k] = cos(theta)
            twImag[k] = sin(theta)
            k += 1
        }
        // Per-stage contiguous twiddle tables
        var stageReal: [Float] = []
        var stageImag: [Float] = []
        var offsets = [Int](repeating: 0, count: m + 1)
        var stage = 1
        while stage <= m {
            offsets[stage] = stageReal.count
            let len = 1 << stage
            let halfLen = len >> 1
            let step = size / len
            var j = 0
            while j < halfLen {
                stageReal.append(twReal[j * step])
                stageImag.append(twImag[j * step])
                j += 1
            }
            stage += 1
        }
        self.stageTwiddleReal = stageReal
        self.stageTwiddleImag = stageImag
        self.stageTwiddleOffsets = offsets
    }
    
    /// In-place forward FFT.
    @inline(__always)
    public func forward(
        real: UnsafeMutablePointer<Float>,
        imag: UnsafeMutablePointer<Float>
    ) {
        var i = 0
        while i < size {
            let rev = bitReversedIndices[i]
            if i < rev {
                let tr = real[i]
                real[i] = real[rev]
                real[rev] = tr
                
                let ti = imag[i]
                imag[i] = imag[rev]
                imag[rev] = ti
            }
            i += 1
        }
        
        // Butterflies: block k outer, element j inner, so real[k+j] is contiguous and SIMD8
        // handles 8 butterflies at once. Butterflies are independent, so the result is
        // bit-identical to the scalar loop order.
        stageTwiddleReal.withUnsafeBufferPointer { stwrBuf in
            stageTwiddleImag.withUnsafeBufferPointer { stwiBuf in
                let stwr = stwrBuf.baseAddress!
                let stwi = stwiBuf.baseAddress!
                var s = 1
                while s <= log2Size {
                    let len = 1 << s
                    let halfLen = len >> 1
                    let twOffset = stageTwiddleOffsets[s]
                    let simdLimit = halfLen - (halfLen % 8)

                    var k = 0
                    while k < size {
                        var j = 0
                        while j < simdLimit {
                            let i1 = k + j
                            let i2 = i1 + halfLen
                            let tw = twOffset + j

                            let wr = SIMD8<Float>(
                                stwr[tw+0], stwr[tw+1], stwr[tw+2], stwr[tw+3],
                                stwr[tw+4], stwr[tw+5], stwr[tw+6], stwr[tw+7]
                            )
                            let wi = SIMD8<Float>(
                                stwi[tw+0], stwi[tw+1], stwi[tw+2], stwi[tw+3],
                                stwi[tw+4], stwi[tw+5], stwi[tw+6], stwi[tw+7]
                            )
                            let ur = SIMD8<Float>(
                                real[i1+0], real[i1+1], real[i1+2], real[i1+3],
                                real[i1+4], real[i1+5], real[i1+6], real[i1+7]
                            )
                            let ui = SIMD8<Float>(
                                imag[i1+0], imag[i1+1], imag[i1+2], imag[i1+3],
                                imag[i1+4], imag[i1+5], imag[i1+6], imag[i1+7]
                            )
                            let vr = SIMD8<Float>(
                                real[i2+0], real[i2+1], real[i2+2], real[i2+3],
                                real[i2+4], real[i2+5], real[i2+6], real[i2+7]
                            )
                            let vi = SIMD8<Float>(
                                imag[i2+0], imag[i2+1], imag[i2+2], imag[i2+3],
                                imag[i2+4], imag[i2+5], imag[i2+6], imag[i2+7]
                            )

                            let tr = (wr * vr) - (wi * vi)
                            let ti = (wr * vi) + (wi * vr)
                            let r1 = ur + tr
                            let m1 = ui + ti
                            let r2 = ur - tr
                            let m2 = ui - ti

                            var lane = 0
                            while lane < 8 {
                                real[i1+lane] = r1[lane]
                                imag[i1+lane] = m1[lane]
                                real[i2+lane] = r2[lane]
                                imag[i2+lane] = m2[lane]
                                lane += 1
                            }
                            j += 8
                        }
                        while j < halfLen {
                            let idx1 = k + j
                            let idx2 = idx1 + halfLen
                            let wr = stwr[twOffset + j]
                            let wi = stwi[twOffset + j]

                            let ur = real[idx1]
                            let ui = imag[idx1]
                            let vr = real[idx2]
                            let vi = imag[idx2]

                            let tr = (wr * vr) - (wi * vi)
                            let ti = (wr * vi) + (wi * vr)

                            real[idx1] = ur + tr
                            imag[idx1] = ui + ti
                            real[idx2] = ur - tr
                            imag[idx2] = ui - ti
                            j += 1
                        }
                        k += len
                    }
                    s += 1
                }
            }
        }
    }
    
    /// Writes |X[k]|^2 to `powerSpectrum[0...halfSize]`.
    @inline(__always)
    public func computePowerSpectrum(
        real: UnsafePointer<Float>,
        imag: UnsafePointer<Float>,
        powerSpectrum: UnsafeMutablePointer<Float>,
        halfSize: Int
    ) {
        let width = 8
        let limit = halfSize - (halfSize % width)
        var i = 0
        
        while i < limit {
            let vr = SIMD8<Float>(
                real[i+0], real[i+1], real[i+2], real[i+3],
                real[i+4], real[i+5], real[i+6], real[i+7]
            )
            let vi = SIMD8<Float>(
                imag[i+0], imag[i+1], imag[i+2], imag[i+3],
                imag[i+4], imag[i+5], imag[i+6], imag[i+7]
            )
            let vPow = (vr * vr) + (vi * vi)
            powerSpectrum[i+0] = vPow[0]
            powerSpectrum[i+1] = vPow[1]
            powerSpectrum[i+2] = vPow[2]
            powerSpectrum[i+3] = vPow[3]
            powerSpectrum[i+4] = vPow[4]
            powerSpectrum[i+5] = vPow[5]
            powerSpectrum[i+6] = vPow[6]
            powerSpectrum[i+7] = vPow[7]
            i += width
        }
        
        while i <= halfSize {
            let r = real[i]
            let im = imag[i]
            powerSpectrum[i] = (r * r) + (im * im)
            i += 1
        }
    }
}
