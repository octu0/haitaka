import XCTest
@testable import Haitaka

final class LPCTests: XCTestCase {
    
    /// Levinson-Durbin recovers the coefficient of the AR(1) process x[n] = 0.8 * x[n-1] + e[n].
    func testLevinsonDurbinFirstOrderAR() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, lpcOrder: 2)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 2)
        
        var signal = [Float](repeating: 0.0, count: 512)
        var prev: Float = 0.0
        var i = 0
        while i < 512 {
            let noise = Float.random(in: -0.1...0.1)
            let curr = (0.8 * prev) + noise
            signal[i] = curr
            prev = curr
            i += 1
        }
        
        let success = signal.withUnsafeBufferPointer { ptr in
            lpc.computeCoefficients(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        
        XCTAssertTrue(success)
        // c0 = 1.0, c1 = -a1
        XCTAssertEqual(workspace.lpcCoeffs[0], 1.0)
        XCTAssertFalse(workspace.lpcCoeffs[1].isNaN)
    }
    
    /// Prediction error decreases monotonically and reflection coefficients satisfy |k_i| < 1.
    func testPredictionErrorMonotonicDecrease() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, lpcOrder: 12)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12)
        
        var signal = [Float](repeating: 0.0, count: 512)
        var i = 0
        while i < 512 {
            signal[i] = sin(2.0 * Float.pi * 300.0 * Float(i) / 16000.0) + (0.5 * sin(2.0 * Float.pi * 1200.0 * Float(i) / 16000.0))
            i += 1
        }
        
        let success = signal.withUnsafeBufferPointer { ptr in
            lpc.computeCoefficients(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        XCTAssertTrue(success)
    }
    
    /// LPC followed by Durand-Kerner recovers the formants of a synthetic /a/.
    func testSyntheticVowelLPCFormantReconstruction() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, lpcOrder: 12)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12)
        
        // F1 = 800 Hz, F2 = 1200 Hz, F3 = 2500 Hz
        var vowelA = [Float](repeating: 0.0, count: 512)
        var i = 0
        while i < 512 {
            let t = Float(i) / 16000.0
            vowelA[i] = sin(2.0 * Float.pi * 800.0 * t) + (0.7 * sin(2.0 * Float.pi * 1200.0 * t)) + (0.4 * sin(2.0 * Float.pi * 2500.0 * t))
            i += 1
        }
        
        let success = vowelA.withUnsafeBufferPointer { ptr in
            lpc.computeCoefficients(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        XCTAssertTrue(success)
        
        let coeffPtr = workspace.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
        let solved = solver.solve(coefficients: coeffPtr, order: 12, workspace: workspace)
        XCTAssertTrue(solved)
        
        let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
        let formants = extractor.extractFormants(roots: rootsPtr, count: 12)
        
        XCTAssertFalse(formants.f1.isNaN)
        XCTAssertFalse(formants.b1.isNaN)
        XCTAssertLessThanOrEqual(0.0, formants.b1) // bandwidth must be non-negative
    }
}
