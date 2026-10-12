import Foundation

public struct DurandKernerSolver: Sendable {
    public init() {}
    
    /// Durand-Kerner root finding for the LPC polynomial, iterating in `workspace.durandKernerCurr/Next`.
    @inline(__always)
    public func solve(
        coefficients: UnsafePointer<Float>,
        order: Int,
        workspace: DSPWorkspace
    ) -> Bool {
        if order < 1 {
            return false
        }
        
        let curr = workspace.durandKernerCurr.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let next = workspace.durandKernerNext.withUnsafeMutableBufferPointer { $0.baseAddress! }
        
        // Initial roots on the unit circle
        let factor = (2.0 * Float.pi) / Float(order)
        var i = 0
        while i < order {
            let theta = (Float(i) * factor) + 0.5
            curr[i] = Complex(real: cos(theta), imag: sin(theta))
            i += 1
        }
        
        var iter = 0
        let maxIter = 80
        let tolerance: Float = 1e-4
        
        while iter < maxIter {
            var maxChange: Float = 0.0
            
            i = 0
            while i < order {
                let zi = curr[i]
                
                // P(z_i) by Horner's method (c_0 = 1)
                var pz = Complex(real: 1.0, imag: 0.0)
                var k = 1
                while k <= order {
                    pz = (pz * zi) + Complex(real: coefficients[k], imag: 0.0)
                    k += 1
                }
                
                // Denominator prod_{j != i} (z_i - z_j)
                var denom = Complex(real: 1.0, imag: 0.0)
                var j = 0
                while j < order {
                    if j != i {
                        denom = denom * (zi - curr[j])
                    }
                    j += 1
                }
                
                let delta = pz / denom
                let newRoot = zi - delta
                next[i] = newRoot
                
                let change = delta.magnitude
                if maxChange < change {
                    maxChange = change
                }
                i += 1
            }
            
            i = 0
            while i < order {
                curr[i] = next[i]
                i += 1
            }
            
            if maxChange < tolerance {
                break
            }
            iter += 1
        }
        
        return true
    }
}

public struct FormantExtractor: Sendable {
    public let sampleRate: Float
    
    public init(sampleRate: Float = 16000.0) {
        self.sampleRate = sampleRate
    }
    
    /// Extracts formants F1-F3 and bandwidths B1-B3 (Hz) from the complex roots.
    @inline(__always)
    public func extractFormants(
        roots: UnsafePointer<Complex>,
        count: Int
    ) -> FormantResult {
        let pi = Float.pi
        var candCount = 0
        
        // Up to 16 candidates
        var candFreq: (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float) = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        var candBw: (Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float, Float) = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        
        var i = 0
        while i < count {
            let z = roots[i]
            if 0.0 < z.imag {
                let r = z.magnitude
                // Resonant poles only: 0.88 <= r < 1.0 (excludes poles outside the unit circle)
                if 0.88 <= r && r < 1.0 {
                    let theta = atan2(z.imag, z.real)
                    let freq = (theta * sampleRate) / (2.0 * pi)
                    let bw = -1.0 * (sampleRate / pi) * log(r)
                    
                    if 250.0 <= freq && freq < 4500.0 {
                        // Insertion sort, ascending
                        withUnsafeMutablePointer(to: &candFreq) { fTuplePtr in
                            let fPtr = UnsafeMutableRawPointer(fTuplePtr).assumingMemoryBound(to: Float.self)
                            withUnsafeMutablePointer(to: &candBw) { bTuplePtr in
                                let bPtr = UnsafeMutableRawPointer(bTuplePtr).assumingMemoryBound(to: Float.self)
                                
                                var insIdx = candCount
                                if 16 <= insIdx {
                                    insIdx = 15
                                }
                                while 0 < insIdx && freq < fPtr[insIdx - 1] {
                                    if insIdx < 16 {
                                        fPtr[insIdx] = fPtr[insIdx - 1]
                                        bPtr[insIdx] = bPtr[insIdx - 1]
                                    }
                                    insIdx -= 1
                                }
                                if insIdx < 16 {
                                    fPtr[insIdx] = freq
                                    bPtr[insIdx] = bw
                                }
                                if candCount < 16 {
                                    candCount += 1
                                }
                            }
                        }
                    }
                }
            }
            i += 1
        }
        
        var f1: Float = 0.0
        var f2: Float = 0.0
        var f3: Float = 0.0
        var b1: Float = 0.0
        var b2: Float = 0.0
        var b3: Float = 0.0
        
        withUnsafePointer(to: candFreq) { fTuplePtr in
            let fPtr = UnsafeRawPointer(fTuplePtr).assumingMemoryBound(to: Float.self)
            withUnsafePointer(to: candBw) { bTuplePtr in
                let bPtr = UnsafeRawPointer(bTuplePtr).assumingMemoryBound(to: Float.self)
                
                if 0 < candCount {
                    f1 = fPtr[0]
                    b1 = bPtr[0]
                }
                if 1 < candCount {
                    f2 = fPtr[1]
                    b2 = bPtr[1]
                }
                if 2 < candCount {
                    f3 = fPtr[2]
                    b3 = bPtr[2]
                }
            }
        }
        
        return FormantResult(f1: f1, f2: f2, f3: f3, b1: b1, b2: b2, b3: b3, count: candCount)
    }
}
