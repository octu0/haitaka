import Foundation

public struct LPC: Sendable {
    public let config: DSPConfig
    
    public init(config: DSPConfig = DSPConfig()) {
        self.config = config
    }
    
    /// Computes the coefficients c_k of the LPC polynomial P(z) = z^P + c1*z^(P-1) + ... + cP.
    @inline(__always)
    public func computeCoefficients(
        ptr: UnsafePointer<Float>,
        count: Int,
        workspace: DSPWorkspace
    ) -> Bool {
        let order = config.lpcOrder
        if count < order + 8 {
            return false
        }
        
        let prePtr = workspace.preemphasizedFrame.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let winPtr = workspace.windowedFrame.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let winTable = workspace.hammingWindow.withUnsafeBufferPointer { $0.baseAddress! }
        let autoCorr = workspace.lpcAutoCorr.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let coeffOut = workspace.lpcCoeffs.withUnsafeMutableBufferPointer { $0.baseAddress! }
        
        // Pre-emphasis: s'[n] = s[n] - 0.97 * s[n-1]
        prePtr[0] = ptr[0]
        let alpha = config.preemphasisCoeff
        var t = 1
        while t < count {
            prePtr[t] = ptr[t] - (alpha * ptr[t-1])
            t += 1
        }
        
        // Hamming window
        VectorOperations.multiply(srcA: prePtr, srcB: winTable, dst: winPtr, count: count)
        
        // Autocorrelation r_0 ... r_P
        var lag = 0
        while lag <= order {
            autoCorr[lag] = VectorOperations.dotProduct(a: winPtr, b: winPtr.advanced(by: lag), count: count - lag)
            lag += 1
        }
        
        let e0 = autoCorr[0]
        if e0 < 1e-10 {
            return false
        }
        
        // Levinson-Durbin recursion (scratch: workspace.lpcTempA)
        let a = workspace.lpcTempA.withUnsafeMutableBufferPointer { $0.baseAddress! }
        a[0] = 1.0
        var e = e0
        
        var i = 1
        while i <= order {
            var s: Float = 0.0
            var j = 1
            while j < i {
                s += a[j] * autoCorr[i - j]
                j += 1
            }
            
            let ki = (autoCorr[i] - s) / e
            a[i] = ki
            
            j = 1
            let limit = i / 2
            while j <= limit {
                let aj = a[j]
                let aij = a[i - j]
                a[j] = aj - (ki * aij)
                a[i - j] = aij - (ki * aj)
                j += 1
            }
            
            e = e * (1.0 - (ki * ki))
            if e < 1e-10 {
                e = 1e-10
            }
            i += 1
        }
        
        // Polynomial coefficients: c_0 = 1, c_k = -a_k
        coeffOut[0] = 1.0
        i = 1
        while i <= order {
            coeffOut[i] = -1.0 * a[i]
            i += 1
        }
        
        return true
    }
}
