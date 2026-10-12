import XCTest
@testable import Haitaka

final class DurandKernerTests: XCTestCase {
    
    func testSolveQuadratic() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: 12)
        
        // P(z) = z^2 - 3z + 2 has roots 1 and 2.
        // Coefficients: [c0 = 1, c1 = -3, c2 = 2].
        let coeffs: [Float] = [1.0, -3.0, 2.0]
        let success = coeffs.withUnsafeBufferPointer { ptr in
            return solver.solve(coefficients: ptr.baseAddress!, order: 2, workspace: workspace)
        }
        
        XCTAssertTrue(success)
        
        let r0 = workspace.durandKernerCurr[0]
        let r1 = workspace.durandKernerCurr[1]
        
        let dist0 = min(
            (r0 - Complex(real: 1.0, imag: 0.0)).magnitude,
            (r0 - Complex(real: 2.0, imag: 0.0)).magnitude
        )
        let dist1 = min(
            (r1 - Complex(real: 1.0, imag: 0.0)).magnitude,
            (r1 - Complex(real: 2.0, imag: 0.0)).magnitude
        )
        
        XCTAssertLessThan(dist0, 1e-3)
        XCTAssertLessThan(dist1, 1e-3)
    }
    
    func testSolve12thOrderLPC() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: 12)
        let pi = Float.pi
        
        // Six conjugate pole pairs at r = 0.92, angles 30, 55, 80, 105, 130, 155 degrees.
        let r: Float = 0.92
        let angles: [Float] = [
            (30.0 * pi) / 180.0,
            (55.0 * pi) / 180.0,
            (80.0 * pi) / 180.0,
            (105.0 * pi) / 180.0,
            (130.0 * pi) / 180.0,
            (155.0 * pi) / 180.0
        ]
        
        // Each pair: (z - r e^{j theta})(z - r e^{-j theta}) = z^2 - (2 r cos theta) z + r^2.
        // Convolve the six quadratics into a 12th-order polynomial.
        var currentPoly: [Float] = [1.0]
        
        var pIdx = 0
        while pIdx < angles.count {
            let theta = angles[pIdx]
            let quad: [Float] = [1.0, -2.0 * r * cos(theta), r * r]
            
            let newDegree = (currentPoly.count - 1) + 2
            var newPoly = [Float](repeating: 0.0, count: newDegree + 1)
            
            var i = 0
            while i < currentPoly.count {
                var j = 0
                while j < quad.count {
                    newPoly[i + j] += currentPoly[i] * quad[j]
                    j += 1
                }
                i += 1
            }
            currentPoly = newPoly
            pIdx += 1
        }
        
        let success = currentPoly.withUnsafeBufferPointer { ptr in
            return solver.solve(coefficients: ptr.baseAddress!, order: 12, workspace: workspace)
        }
        
        XCTAssertTrue(success)
        
        // Every root magnitude should be close to 0.92.
        var idx = 0
        while idx < 12 {
            let root = workspace.durandKernerCurr[idx]
            XCTAssertLessThan(abs(root.magnitude - r), 0.05)
            idx += 1
        }
    }
    
    func testExtractFormantsVowelA() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // Vowel /a/: F1 = 800 Hz, F2 = 1200 Hz, F3 = 2500 Hz.
        let targetF1: Float = 800.0
        let targetF2: Float = 1200.0
        let targetF3: Float = 2500.0
        let r: Float = 0.96
        
        let theta1 = (2.0 * pi * targetF1) / 16000.0
        let theta2 = (2.0 * pi * targetF2) / 16000.0
        let theta3 = (2.0 * pi * targetF3) / 16000.0
        
        let roots: [Complex] = [
            Complex(real: r * cos(theta1), imag: r * sin(theta1)),
            Complex(real: r * cos(theta1), imag: -r * sin(theta1)),
            Complex(real: r * cos(theta2), imag: r * sin(theta2)),
            Complex(real: r * cos(theta2), imag: -r * sin(theta2)),
            Complex(real: r * cos(theta3), imag: r * sin(theta3)),
            Complex(real: r * cos(theta3), imag: -r * sin(theta3))
        ]
        
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        XCTAssertLessThan(abs(result.f1 - targetF1), 1.0)
        XCTAssertLessThan(abs(result.f2 - targetF2), 1.0)
        XCTAssertLessThan(abs(result.f3 - targetF3), 1.0)
    }
    
    func testExtractFormantsVowelI() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // Vowel /i/: F1 = 300 Hz, F2 = 2300 Hz, F3 = 3000 Hz.
        let targetF1: Float = 300.0
        let targetF2: Float = 2300.0
        let targetF3: Float = 3000.0
        let r: Float = 0.96
        
        let theta1 = (2.0 * pi * targetF1) / 16000.0
        let theta2 = (2.0 * pi * targetF2) / 16000.0
        let theta3 = (2.0 * pi * targetF3) / 16000.0
        
        let roots: [Complex] = [
            Complex(real: r * cos(theta1), imag: r * sin(theta1)),
            Complex(real: r * cos(theta1), imag: -r * sin(theta1)),
            Complex(real: r * cos(theta2), imag: r * sin(theta2)),
            Complex(real: r * cos(theta2), imag: -r * sin(theta2)),
            Complex(real: r * cos(theta3), imag: r * sin(theta3)),
            Complex(real: r * cos(theta3), imag: -r * sin(theta3))
        ]
        
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        XCTAssertLessThan(abs(result.f1 - targetF1), 1.0)
        XCTAssertLessThan(abs(result.f2 - targetF2), 1.0)
        XCTAssertLessThan(abs(result.f3 - targetF3), 1.0)
    }
    
    func testFilterOutPolesOutsideUnitCircle() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // Pole radius 0.70 is below the 0.88 resonance threshold.
        let freq: Float = 1000.0
        let theta = (2.0 * pi * freq) / 16000.0
        let weakRoot = Complex(real: 0.70 * cos(theta), imag: 0.70 * sin(theta))
        
        let roots: [Complex] = [weakRoot]
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        XCTAssertEqual(result.count, 0)
        XCTAssertEqual(result.f1, 0.0)
    }
    
    func testBandwidthCalculation() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        let r: Float = 0.96
        let freq: Float = 1000.0
        let theta = (2.0 * pi * freq) / 16000.0
        let root = Complex(real: r * cos(theta), imag: r * sin(theta))
        
        let roots: [Complex] = [root]
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        let expectedBw = -1.0 * (16000.0 / pi) * log(r)
        XCTAssertLessThan(abs(result.b1 - expectedBw), 1.0)
    }
    
    func testFilterOutPolesOnOrOutsideUnitCircle() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // Outside the unit circle (r = 1.15).
        let freq: Float = 1000.0
        let theta = (2.0 * pi * freq) / 16000.0
        let divergingRoot = Complex(real: 1.15 * cos(theta), imag: 1.15 * sin(theta))
        
        // On the unit circle (r = 1.00).
        let unitRoot = Complex(real: 1.00 * cos(theta), imag: 1.00 * sin(theta))
        
        let roots: [Complex] = [divergingRoot, unitRoot]
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        // Poles with r >= 1 are rejected.
        XCTAssertEqual(result.count, 0)
        XCTAssertEqual(result.f1, 0.0)
        XCTAssertEqual(result.b1, 0.0)
    }
}
