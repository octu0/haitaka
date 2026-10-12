import XCTest
@testable import Haitaka

/// Adversarial and stress tests for the audio DSP front end.
final class M1ChallengerStressTests: XCTestCase {
    
    // MARK: - Malformed WAV
    
    private func buildCustomWav(
        riffHeader: [UInt8] = [0x52, 0x49, 0x46, 0x46],
        riffSize: UInt32 = 36 + 640,
        waveHeader: [UInt8] = [0x57, 0x41, 0x56, 0x45],
        fmtHeader: [UInt8] = [0x66, 0x6d, 0x74, 0x20],
        fmtSize: UInt32 = 16,
        audioFormat: UInt16 = 1,
        channels: UInt16 = 1,
        sampleRate: UInt32 = 16000,
        bitsPerSample: UInt16 = 16,
        dataHeader: [UInt8] = [0x64, 0x61, 0x74, 0x61],
        dataSize: UInt32 = 640,
        rawPayload: [UInt8] = [UInt8](repeating: 0, count: 640)
    ) -> [UInt8] {
        var bytes: [UInt8] = []
        
        bytes.append(contentsOf: riffHeader)
        bytes.append(UInt8(riffSize & 0xFF))
        bytes.append(UInt8((riffSize >> 8) & 0xFF))
        bytes.append(UInt8((riffSize >> 16) & 0xFF))
        bytes.append(UInt8((riffSize >> 24) & 0xFF))
        bytes.append(contentsOf: waveHeader)
        
        bytes.append(contentsOf: fmtHeader)
        bytes.append(UInt8(fmtSize & 0xFF))
        bytes.append(UInt8((fmtSize >> 8) & 0xFF))
        bytes.append(UInt8((fmtSize >> 16) & 0xFF))
        bytes.append(UInt8((fmtSize >> 24) & 0xFF))
        
        bytes.append(UInt8(audioFormat & 0xFF))
        bytes.append(UInt8((audioFormat >> 8) & 0xFF))
        bytes.append(UInt8(channels & 0xFF))
        bytes.append(UInt8((channels >> 8) & 0xFF))
        bytes.append(UInt8(sampleRate & 0xFF))
        bytes.append(UInt8((sampleRate >> 8) & 0xFF))
        bytes.append(UInt8((sampleRate >> 16) & 0xFF))
        bytes.append(UInt8((sampleRate >> 24) & 0xFF))
        
        let byteRate = (sampleRate * UInt32(channels) * UInt32(bitsPerSample)) / 8
        bytes.append(UInt8(byteRate & 0xFF))
        bytes.append(UInt8((byteRate >> 8) & 0xFF))
        bytes.append(UInt8((byteRate >> 16) & 0xFF))
        bytes.append(UInt8((byteRate >> 24) & 0xFF))
        
        let blockAlign = (UInt32(channels) * UInt32(bitsPerSample)) / 8
        bytes.append(UInt8(blockAlign & 0xFF))
        bytes.append(UInt8((blockAlign >> 8) & 0xFF))
        bytes.append(UInt8(bitsPerSample & 0xFF))
        bytes.append(UInt8((bitsPerSample >> 8) & 0xFF))
        
        bytes.append(contentsOf: dataHeader)
        bytes.append(UInt8(dataSize & 0xFF))
        bytes.append(UInt8((dataSize >> 8) & 0xFF))
        bytes.append(UInt8((dataSize >> 16) & 0xFF))
        bytes.append(UInt8((dataSize >> 24) & 0xFF))
        
        bytes.append(contentsOf: rawPayload)
        return bytes
    }
    
    /// Empty and truncated inputs throw instead of crashing.
    func testWavParserUnderflow() {
        let parser = WavParser()
        
        XCTAssertThrowsError(try parser.parse(bytes: [])) { error in
            XCTAssertEqual(error as? WavParserError, .invalidHeader)
        }
        
        var len = 1
        while len <= 43 {
            let partial = [UInt8](repeating: 0x55, count: len)
            XCTAssertThrowsError(try parser.parse(bytes: partial))
            len += 1
        }
    }
    
    /// A data chunk claiming 0xFFFFFFFF bytes is rejected.
    func testWavParserOversizedChunkSafety() {
        let parser = WavParser()
        
        let maliciousBytes = buildCustomWav(dataSize: 0xFFFFFFFF, rawPayload: [0, 1, 2, 3])
        XCTAssertThrowsError(try parser.parse(bytes: maliciousBytes)) { error in
            XCTAssertEqual(error as? WavParserError, .invalidHeader)
        }
    }
    
    /// Unknown chunks (JUNK, LIST) before fmt are skipped.
    func testWavParserWithJunkAndListChunks() throws {
        let parser = WavParser()
        var bytes: [UInt8] = []
        
        bytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        let totalSize: UInt32 = 36 + 12 + 640
        bytes.append(UInt8(totalSize & 0xFF))
        bytes.append(UInt8((totalSize >> 8) & 0xFF))
        bytes.append(UInt8((totalSize >> 16) & 0xFF))
        bytes.append(UInt8((totalSize >> 24) & 0xFF))
        bytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"
        
        bytes.append(contentsOf: [0x4a, 0x55, 0x4e, 0x4b]) // "JUNK", 4 data bytes
        bytes.append(contentsOf: [4, 0, 0, 0])
        bytes.append(contentsOf: [0xDE, 0xAD, 0xBE, 0xEF])
        
        bytes.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])
        bytes.append(contentsOf: [16, 0, 0, 0])
        bytes.append(contentsOf: [1, 0]) // PCM
        bytes.append(contentsOf: [1, 0]) // mono
        bytes.append(contentsOf: [0x80, 0x3E, 0, 0]) // 16000 Hz
        bytes.append(contentsOf: [0x00, 0x7D, 0, 0]) // byte rate 32000
        bytes.append(contentsOf: [2, 0]) // block align 2
        bytes.append(contentsOf: [16, 0]) // 16 bits
        
        bytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        bytes.append(contentsOf: [16, 0, 0, 0]) // 8 samples
        var s = 0
        while s < 8 {
            bytes.append(0)
            bytes.append(0x10) // 4096 as Int16
            s += 1
        }
        
        let result = try parser.parse(bytes: bytes)
        XCTAssertEqual(result.sampleRate, 16000)
        XCTAssertEqual(result.channels, 1)
        XCTAssertEqual(result.pcmData.count, 8)
        XCTAssertEqual(result.pcmData[0], 4096.0 / 32768.0)
    }
    
    /// Unsupported formats (A-law, mu-law, 8-bit PCM, non-32-bit float) are rejected.
    func testWavParserUnsupportedFormats() {
        let parser = WavParser()

        let alawWav = buildCustomWav(audioFormat: 6, bitsPerSample: 8)
        XCTAssertThrowsError(try parser.parse(bytes: alawWav))

        let mulawWav = buildCustomWav(audioFormat: 7, bitsPerSample: 8)
        XCTAssertThrowsError(try parser.parse(bytes: mulawWav))

        let pcm8Wav = buildCustomWav(audioFormat: 1, bitsPerSample: 8)
        XCTAssertThrowsError(try parser.parse(bytes: pcm8Wav))

        // Float format code with a bit depth other than 32.
        let badFloatWav = buildCustomWav(audioFormat: 3, bitsPerSample: 16)
        XCTAssertThrowsError(try parser.parse(bytes: badFloatWav))
    }

    /// Formats seen in real recordings (32-bit float, 24/32-bit integer) are decoded.
    func testWavParserWideFormats() throws {
        let parser = WavParser()

        // 32-bit IEEE float: four samples of 0.5.
        var floatPayload: [UInt8] = []
        var i = 0
        while i < 4 {
            let bits = Float(0.5).bitPattern
            floatPayload.append(UInt8(bits & 0xFF))
            floatPayload.append(UInt8((bits >> 8) & 0xFF))
            floatPayload.append(UInt8((bits >> 16) & 0xFF))
            floatPayload.append(UInt8((bits >> 24) & 0xFF))
            i += 1
        }
        let floatWav = buildCustomWav(
            audioFormat: 3, bitsPerSample: 32,
            dataSize: UInt32(floatPayload.count), rawPayload: floatPayload)
        let floatResult = try parser.parse(bytes: floatWav)
        XCTAssertEqual(floatResult.pcmData.count, 4)
        XCTAssertEqual(floatResult.pcmData[0], 0.5, accuracy: 1e-6)

        // 32-bit integer: upper 16 bits hold 4096, lower bits zero.
        var int32Payload: [UInt8] = []
        i = 0
        while i < 4 {
            int32Payload.append(contentsOf: [0, 0, 0x00, 0x10])
            i += 1
        }
        let int32Wav = buildCustomWav(
            audioFormat: 1, bitsPerSample: 32,
            dataSize: UInt32(int32Payload.count), rawPayload: int32Payload)
        let int32Result = try parser.parse(bytes: int32Wav)
        XCTAssertEqual(int32Result.pcmData.count, 4)
        XCTAssertEqual(int32Result.pcmData[0], 4096.0 / 32768.0, accuracy: 1e-6)

        // 24-bit integer
        var int24Payload: [UInt8] = []
        i = 0
        while i < 4 {
            int24Payload.append(contentsOf: [0, 0x00, 0x10])
            i += 1
        }
        let int24Wav = buildCustomWav(
            audioFormat: 1, bitsPerSample: 24,
            dataSize: UInt32(int24Payload.count), rawPayload: int24Payload)
        let int24Result = try parser.parse(bytes: int24Wav)
        XCTAssertEqual(int24Result.pcmData.count, 4)
        XCTAssertEqual(int24Result.pcmData[0], 4096.0 / 32768.0, accuracy: 1e-6)
    }

    /// A chunk after an odd-sized chunk is found.
    /// RIFF pads odd-sized chunks with one byte to keep word alignment.
    func testWavParserOddSizedChunkPadding() throws {
        let parser = WavParser()
        var bytes: [UInt8] = []

        func appendUInt32(_ v: UInt32) {
            bytes.append(UInt8(v & 0xFF))
            bytes.append(UInt8((v >> 8) & 0xFF))
            bytes.append(UInt8((v >> 16) & 0xFF))
            bytes.append(UInt8((v >> 24) & 0xFF))
        }

        // Leading odd-sized unknown chunk: 3 bytes plus 1 pad byte.
        bytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        appendUInt32(0)  // RIFF size patched later
        bytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45])

        bytes.append(contentsOf: [0x62, 0x65, 0x78, 0x74])  // "bext"
        appendUInt32(3)
        bytes.append(contentsOf: [1, 2, 3, 0])  // 3 bytes + pad

        bytes.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])  // "fmt "
        appendUInt32(16)
        bytes.append(contentsOf: [1, 0])
        bytes.append(contentsOf: [1, 0])
        bytes.append(contentsOf: [0x80, 0x3E, 0, 0])
        bytes.append(contentsOf: [0x00, 0x7D, 0, 0])
        bytes.append(contentsOf: [2, 0])
        bytes.append(contentsOf: [16, 0])

        bytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])  // "data"
        appendUInt32(8)
        var s = 0
        while s < 4 {
            bytes.append(0)
            bytes.append(0x10)
            s += 1
        }

        let riffSize = UInt32(bytes.count - 8)
        bytes[4] = UInt8(riffSize & 0xFF)
        bytes[5] = UInt8((riffSize >> 8) & 0xFF)
        bytes[6] = UInt8((riffSize >> 16) & 0xFF)
        bytes[7] = UInt8((riffSize >> 24) & 0xFF)

        let result = try parser.parse(bytes: bytes)
        XCTAssertEqual(result.sampleRate, 16000)
        XCTAssertEqual(result.pcmData.count, 4)
        XCTAssertEqual(result.pcmData[0], 4096.0 / 32768.0, accuracy: 1e-6)
    }
    
    // MARK: - Degenerate and extreme signals
    
    /// Digital silence keeps VAD, pitch, LPC, formant, and filterbank outputs finite and inactive.
    func testSilencePipelineStability() {
        let workspace = DSPWorkspace()
        let frameSize = 320
        let zeroBuffer = [Float](repeating: 0.0, count: frameSize)
        
        let vad = VAD()
        let pitchDetector = PitchDetector()
        let lpc = LPC()
        let formantExtractor = FormantExtractor()
        let filterbank = Filterbank()
        
        zeroBuffer.withUnsafeBufferPointer { ptr in
            let base = ptr.baseAddress!
            
            let vadRes = vad.processFrame(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertFalse(vadRes.isSpeech)
            XCTAssertEqual(vadRes.rms, 0.0)
            XCTAssertEqual(vadRes.zcr, 0.0)
            XCTAssertEqual(vadRes.voicingRatio, 0.0)
            
            let pitchRes = pitchDetector.detectPitch(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertFalse(pitchRes.isVoiced)
            XCTAssertEqual(pitchRes.f0, 0.0)
            XCTAssertEqual(pitchRes.hnr, 0.0)
            
            let lpcSuccess = lpc.computeCoefficients(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertFalse(lpcSuccess)
            
            let roots = [Complex](repeating: Complex(real: 0.0, imag: 0.0), count: 12)
            let formantRes = roots.withUnsafeBufferPointer { rPtr in
                formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: 12)
            }
            XCTAssertEqual(formantRes.count, 0)
            XCTAssertEqual(formantRes.f1, 0.0)
            
            let features = filterbank.extractFeatures(
                pcmPtr: base,
                count: frameSize,
                workspace: workspace
            )
            XCTAssertEqual(features.count, 64)
            for f in features {
                XCTAssertFalse(f.isNaN)
                XCTAssertFalse(f.isInfinite)
                XCTAssertLessThanOrEqual(0.0, f)
                XCTAssertLessThanOrEqual(f, 1.0)
            }
        }
    }
    
    /// A constant DC input causes no division by zero or divergence.
    func testDCOffsetPipelineStability() {
        let workspace = DSPWorkspace()
        let frameSize = 512
        let dcBuffer = [Float](repeating: 1.0, count: frameSize)
        
        let pitchDetector = PitchDetector()
        let lpc = LPC()
        let dkSolver = DurandKernerSolver()
        let formantExtractor = FormantExtractor()
        let filterbank = Filterbank()
        
        dcBuffer.withUnsafeBufferPointer { ptr in
            let base = ptr.baseAddress!
            
            // A constant has no periodicity, so it must not be voiced.
            let pitchRes = pitchDetector.detectPitch(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertFalse(pitchRes.isVoiced)
            
            // Pre-emphasis leaves a small constant (1.0 - 0.97 * 1.0 = 0.03), so LPC succeeds.
            let lpcSuccess = lpc.computeCoefficients(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertTrue(lpcSuccess)
            
            let coeffPtr = workspace.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
            let solved = dkSolver.solve(coefficients: coeffPtr, order: 12, workspace: workspace)
            XCTAssertTrue(solved)
            
            let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
            let formantRes = formantExtractor.extractFormants(roots: rootsPtr, count: 12)
            
            let features = filterbank.extractFeatures(
                pcmPtr: base,
                count: frameSize,
                workspace: workspace
            )
            XCTAssertEqual(features.count, 64)
            for f in features {
                XCTAssertFalse(f.isNaN)
                XCTAssertFalse(f.isInfinite)
                XCTAssertLessThanOrEqual(0.0, f)
                XCTAssertLessThanOrEqual(f, 1.0)
            }
        }
    }
    
    /// Very large (1e6) and very small (1e-6, 1e-15) amplitudes cause no overflow or underflow.
    func testExtremeAmplitudeStability() {
        let workspace = DSPWorkspace()
        let frameSize = 512
        let lpc = LPC()
        let dkSolver = DurandKernerSolver()
        let formantExtractor = FormantExtractor()
        let filterbank = Filterbank()
        let pitchDetector = PitchDetector()
        
        let amplitudes: [Float] = [1e6, 1e-6, 1e-15]
        
        for amp in amplitudes {
            var buffer = [Float](repeating: 0.0, count: frameSize)
            var i = 0
            while i < frameSize {
                buffer[i] = amp * sin(2.0 * Float.pi * 200.0 * Float(i) / 16000.0)
                i += 1
            }
            
            buffer.withUnsafeBufferPointer { ptr in
                let base = ptr.baseAddress!
                
                let pitchRes = pitchDetector.detectPitch(ptr: base, count: frameSize, workspace: workspace)
                XCTAssertFalse(pitchRes.f0.isNaN)
                XCTAssertFalse(pitchRes.hnr.isNaN)
                
                let lpcSuccess = lpc.computeCoefficients(ptr: base, count: frameSize, workspace: workspace)
                if lpcSuccess {
                    let coeffPtr = workspace.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
                    let solved = dkSolver.solve(coefficients: coeffPtr, order: 12, workspace: workspace)
                    XCTAssertTrue(solved)
                    
                    let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
                    let formantRes = formantExtractor.extractFormants(roots: rootsPtr, count: 12)
                    XCTAssertFalse(formantRes.f1.isNaN)
                    
                    let features = filterbank.extractFeatures(
                        pcmPtr: base,
                        count: frameSize,
                        workspace: workspace
                    )
                    XCTAssertEqual(features.count, 64)
                    for f in features {
                        XCTAssertFalse(f.isNaN)
                        XCTAssertFalse(f.isInfinite)
                        XCTAssertLessThanOrEqual(0.0, f)
                        XCTAssertLessThanOrEqual(f, 1.0)
                    }
                }
            }
        }
    }
    
    /// Tones near Nyquist (7999 Hz) and far below the pitch range (10 Hz).
    func testBoundaryFrequencies() {
        let workspace = DSPWorkspace()
        let frameSize = 512
        let pitchDetector = PitchDetector()
        let vad = VAD()
        
        var highFreq = [Float](repeating: 0.0, count: frameSize)
        var i = 0
        while i < frameSize {
            highFreq[i] = 0.8 * sin(2.0 * Float.pi * 7999.0 * Float(i) / 16000.0)
            i += 1
        }
        
        highFreq.withUnsafeBufferPointer { ptr in
            let res = pitchDetector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
            // 7999 Hz is far above the voice range; only require a finite result.
            XCTAssertFalse(res.f0.isNaN)
            
            let vadRes = vad.processFrame(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertFalse(vadRes.rms.isNaN)
        }
        
        var lowFreq = [Float](repeating: 0.0, count: frameSize)
        i = 0
        while i < frameSize {
            lowFreq[i] = 0.8 * sin(2.0 * Float.pi * 10.0 * Float(i) / 16000.0)
            i += 1
        }
        
        lowFreq.withUnsafeBufferPointer { ptr in
            let res = pitchDetector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
            // 10 Hz is below the 50-500 Hz pitch search range.
            XCTAssertFalse(res.isVoiced)
        }
    }
    
    /// A hard-clipped (square) high-amplitude signal stays stable.
    func testHardClippedSignalStability() {
        let workspace = DSPWorkspace()
        let frameSize = 512
        var clippedBuffer = [Float](repeating: 0.0, count: frameSize)
        
        // 200 Hz square wave clipped to +/-1.0.
        var i = 0
        while i < frameSize {
            let s = sin(2.0 * Float.pi * 200.0 * Float(i) / 16000.0)
            if 0.0 <= s {
                clippedBuffer[i] = 1.0
            }
            if s < 0.0 {
                clippedBuffer[i] = -1.0
            }
            i += 1
        }
        
        let pitchDetector = PitchDetector()
        let lpc = LPC()
        let dkSolver = DurandKernerSolver()
        let formantExtractor = FormantExtractor()
        let filterbank = Filterbank()
        
        clippedBuffer.withUnsafeBufferPointer { ptr in
            let base = ptr.baseAddress!
            let pitchRes = pitchDetector.detectPitch(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertTrue(pitchRes.isVoiced)
            XCTAssertLessThan(abs(pitchRes.f0 - 200.0), 10.0)
            
            let lpcSuccess = lpc.computeCoefficients(ptr: base, count: frameSize, workspace: workspace)
            XCTAssertTrue(lpcSuccess)
            
            let coeffPtr = workspace.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
            let solved = dkSolver.solve(coefficients: coeffPtr, order: 12, workspace: workspace)
            XCTAssertTrue(solved)
            
            let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
            let formantRes = formantExtractor.extractFormants(roots: rootsPtr, count: 12)
            
            let features = filterbank.extractFeatures(
                pcmPtr: base,
                count: frameSize,
                workspace: workspace
            )
            XCTAssertEqual(features.count, 64)
            for f in features {
                XCTAssertFalse(f.isNaN)
                XCTAssertLessThanOrEqual(0.0, f)
                XCTAssertLessThanOrEqual(f, 1.0)
            }
        }
    }
    
    /// NaN and Inf in the input do not crash any DSP stage.
    func testNaNAndInfInputSafety() {
        let workspace = DSPWorkspace(melChannels: 64)
        let frameSize = 512
        var corruptBuffer = [Float](repeating: 0.0, count: frameSize)
        corruptBuffer[10] = Float.nan
        corruptBuffer[20] = Float.infinity
        corruptBuffer[30] = -Float.infinity
        
        let vad = VAD()
        let pitchDetector = PitchDetector()
        let lpc = LPC()
        let dkSolver = DurandKernerSolver()
        let formantExtractor = FormantExtractor()
        let filterbank = Filterbank()
        
        corruptBuffer.withUnsafeBufferPointer { ptr in
            let base = ptr.baseAddress!
            
            let vadRes = vad.processFrame(ptr: base, count: frameSize, workspace: workspace)
            _ = vadRes.isSpeech
            
            let pitchRes = pitchDetector.detectPitch(ptr: base, count: frameSize, workspace: workspace)
            _ = pitchRes.isVoiced
            
            _ = lpc.computeCoefficients(ptr: base, count: frameSize, workspace: workspace)
            
            var nanCoeffs = [Float](repeating: Float.nan, count: 13)
            nanCoeffs[0] = 1.0
            nanCoeffs.withUnsafeBufferPointer { cPtr in
                _ = dkSolver.solve(coefficients: cPtr.baseAddress!, order: 12, workspace: workspace)
            }
            
            var nanRoots = [Complex](repeating: Complex(real: Float.nan, imag: Float.infinity), count: 12)
            nanRoots.withUnsafeBufferPointer { rPtr in
                let formantRes = formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: 12)
                XCTAssertEqual(formantRes.count, 0)
            }
            
            let features = filterbank.extractFeatures(
                pcmPtr: base,
                count: frameSize,
                workspace: workspace
            )
            XCTAssertEqual(features.count, 64)
        }
    }
    
    // MARK: - Durand-Kerner ill-conditioned polynomials
    
    /// The 12-fold root of (z - 0.5)^12 avoids division by zero and terminates.
    func testDurandKernerMultipleRoots() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace()
        
        // Binomial expansion coefficients of (z - 0.5)^12.
        var coeffs = [Float](repeating: 0.0, count: 13)
        coeffs[0] = 1.0
        
        func comb(_ n: Int, _ k: Int) -> Float {
            if k == 0 || k == n {
                return 1.0
            }
            var res: Float = 1.0
            var i = 1
            while i <= k {
                res = res * Float(n - i + 1) / Float(i)
                i += 1
            }
            return res
        }
        
        var k = 1
        while k <= 12 {
            let sign: Float = (k % 2 == 0) ? 1.0 : -1.0
            let p = pow(0.5, Float(k))
            coeffs[k] = comb(12, k) * sign * p
            k += 1
        }
        
        coeffs.withUnsafeBufferPointer { cPtr in
            let solved = solver.solve(coefficients: cPtr.baseAddress!, order: 12, workspace: workspace)
            XCTAssertTrue(solved)
            
            var idx = 0
            while idx < 12 {
                let r = workspace.durandKernerCurr[idx]
                XCTAssertFalse(r.real.isNaN)
                XCTAssertFalse(r.imag.isNaN)
                XCTAssertFalse(r.real.isInfinite)
                XCTAssertFalse(r.imag.isInfinite)
                idx += 1
            }
        }
    }
    
    /// P(z) = z^12 (all lower coefficients zero) avoids division by zero and terminates.
    func testDurandKernerAllZeroCoefficients() {
        let solver = DurandKernerSolver()
        let extractor = FormantExtractor()
        let workspace = DSPWorkspace()
        
        var coeffs = [Float](repeating: 0.0, count: 13)
        coeffs[0] = 1.0
        
        coeffs.withUnsafeBufferPointer { cPtr in
            let solved = solver.solve(coefficients: cPtr.baseAddress!, order: 12, workspace: workspace)
            XCTAssertTrue(solved)
            
            var idx = 0
            while idx < 12 {
                let r = workspace.durandKernerCurr[idx]
                XCTAssertFalse(r.real.isNaN)
                XCTAssertFalse(r.imag.isNaN)
                // Roots converge toward the origin.
                XCTAssertLessThan(r.magnitude, 0.5)
                idx += 1
            }
            
            // All roots are below the resonance radius threshold (0.88), so no formants are reported.
            let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
            let formantRes = extractor.extractFormants(roots: rootsPtr, count: 12)
            XCTAssertEqual(formantRes.count, 0)
            XCTAssertEqual(formantRes.f1, 0.0)
        }
    }
    
    /// Very large (1e10) and very small (1e-10) coefficients do not crash.
    func testDurandKernerIllConditionedCoefficients() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace()
        
        var hugeCoeffs = [Float](repeating: 1e10, count: 13)
        hugeCoeffs[0] = 1.0
        hugeCoeffs.withUnsafeBufferPointer { cPtr in
            let solved = solver.solve(coefficients: cPtr.baseAddress!, order: 12, workspace: workspace)
            XCTAssertTrue(solved)
        }
        
        var tinyCoeffs = [Float](repeating: 1e-10, count: 13)
        tinyCoeffs[0] = 1.0
        tinyCoeffs.withUnsafeBufferPointer { cPtr in
            let solved = solver.solve(coefficients: cPtr.baseAddress!, order: 12, workspace: workspace)
            XCTAssertTrue(solved)
        }
    }
    
    /// Invalid orders (0, -1) are handled safely.
    func testDurandKernerInvalidOrder() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace()
        var coeffs: [Float] = [1.0, 0.5]
        
        coeffs.withUnsafeBufferPointer { cPtr in
            let r0 = solver.solve(coefficients: cPtr.baseAddress!, order: 0, workspace: workspace)
            XCTAssertFalse(r0)
            
            let rNeg = solver.solve(coefficients: cPtr.baseAddress!, order: -5, workspace: workspace)
            XCTAssertFalse(rNeg)
        }
    }
    
    // MARK: - SIMD boundary lengths
    
    /// Vector operations are correct at lengths around SIMD width multiples.
    func testVectorOperationsBoundaryCounts() {
        let counts = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 319, 320]
        
        for count in counts {
            var srcA = [Float](repeating: 1.5, count: max(count, 1))
            var srcB = [Float](repeating: 2.0, count: max(count, 1))
            var dst = [Float](repeating: 0.0, count: max(count, 1))
            
            srcA.withUnsafeBufferPointer { aPtr in
                srcB.withUnsafeBufferPointer { bPtr in
                    dst.withUnsafeMutableBufferPointer { dPtr in
                        if 0 < count {
                            let dot = VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: count)
                            let expectedDot = Float(count) * 3.0
                            XCTAssertLessThan(abs(dot - expectedDot), 1e-3)
                            
                            let sumSq = VectorOperations.sumOfSquares(ptr: aPtr.baseAddress!, count: count)
                            let expectedSumSq = Float(count) * 2.25
                            XCTAssertLessThan(abs(sumSq - expectedSumSq), 1e-3)
                            
                            let maxMag = VectorOperations.maxMagnitude(ptr: aPtr.baseAddress!, count: count)
                            XCTAssertEqual(maxMag, 1.5)
                            
                            VectorOperations.multiply(srcA: aPtr.baseAddress!, srcB: bPtr.baseAddress!, dst: dPtr.baseAddress!, count: count)
                            var k = 0
                            while k < count {
                                XCTAssertEqual(dPtr[k], 3.0)
                                k += 1
                            }
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Long-running streaming
    
    /// 10,000 frames (about 200 s) of continuous processing without state corruption.
    func testLongRunningStreamStability() {
        let workspace = DSPWorkspace()
        let vad = VAD()
        let pitchDetector = PitchDetector()
        let lpc = LPC()
        let dkSolver = DurandKernerSolver()
        let formantExtractor = FormantExtractor()
        let filterbank = Filterbank()
        
        let frameSize = 512
        var dummyFrame = [Float](repeating: 0.0, count: frameSize)
        
        var frameIndex = 0
        let totalFrames = 10000
        
        dummyFrame.withUnsafeBufferPointer { ptr in
            let base = ptr.baseAddress!
            
            while frameIndex < totalFrames {
                // Alternate between tone and silence.
                let isSignal = (frameIndex % 50) < 30
                if isSignal {
                    var s = 0
                    while s < frameSize {
                        dummyFrame[s] = 0.5 * sin(2.0 * Float.pi * 250.0 * Float(frameIndex * 160 + s) / 16000.0)
                        s += 1
                    }
                }
                if isSignal != true {
                    var s = 0
                    while s < frameSize {
                        dummyFrame[s] = 0.0001 * Float.random(in: -1.0...1.0)
                        s += 1
                    }
                }
                
                let vadRes = vad.processFrame(ptr: base, count: frameSize, workspace: workspace)
                XCTAssertFalse(vadRes.rms.isNaN)
                XCTAssertFalse(vadRes.noiseFloor.isNaN)
                
                let pitchRes = pitchDetector.detectPitch(ptr: base, count: frameSize, workspace: workspace)
                XCTAssertFalse(pitchRes.f0.isNaN)
                
                let lpcSuccess = lpc.computeCoefficients(ptr: base, count: frameSize, workspace: workspace)
                if lpcSuccess {
                    let coeffPtr = workspace.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
                    _ = dkSolver.solve(coefficients: coeffPtr, order: 12, workspace: workspace)
                    let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
                    let formantRes = formantExtractor.extractFormants(roots: rootsPtr, count: 12)
                    
                    let features = filterbank.extractFeatures(
                        pcmPtr: base,
                        count: frameSize,
                        workspace: workspace
                    )
                    XCTAssertEqual(features.count, 64)
                }
                
                frameIndex += 1
            }
        }
    }
    
    // MARK: - FFT extreme inputs
    
    /// FFT stays numerically stable on zero, impulse, Nyquist, constant, and subnormal inputs.
    func testFFTExtremeInputsStability() {
        let fft = FFT(size: 512)
        var real = [Float](repeating: 0.0, count: 512)
        var imag = [Float](repeating: 0.0, count: 512)
        var power = [Float](repeating: 0.0, count: 257)
        
        // All zeros.
        real.withUnsafeMutableBufferPointer { rBuf in
            imag.withUnsafeMutableBufferPointer { iBuf in
                power.withUnsafeMutableBufferPointer { pBuf in
                    let rPtr = rBuf.baseAddress!
                    let iPtr = iBuf.baseAddress!
                    let pPtr = pBuf.baseAddress!
                    
                    fft.forward(real: rPtr, imag: iPtr)
                    fft.computePowerSpectrum(real: rPtr, imag: iPtr, powerSpectrum: pPtr, halfSize: 256)
                    
                    var k = 0
                    while k <= 256 {
                        XCTAssertEqual(rPtr[k], 0.0)
                        XCTAssertEqual(iPtr[k], 0.0)
                        XCTAssertEqual(pPtr[k], 0.0)
                        XCTAssertFalse(pPtr[k].isNaN)
                        XCTAssertFalse(pPtr[k].isInfinite)
                        k += 1
                    }
                }
            }
        }
        
        // Unit impulse: every bin has real 1, imag 0, power 1.
        var kInit = 0
        while kInit < 512 {
            real[kInit] = 0.0
            imag[kInit] = 0.0
            kInit += 1
        }
        real[0] = 1.0
        
        real.withUnsafeMutableBufferPointer { rBuf in
            imag.withUnsafeMutableBufferPointer { iBuf in
                power.withUnsafeMutableBufferPointer { pBuf in
                    let rPtr = rBuf.baseAddress!
                    let iPtr = iBuf.baseAddress!
                    let pPtr = pBuf.baseAddress!
                    
                    fft.forward(real: rPtr, imag: iPtr)
                    fft.computePowerSpectrum(real: rPtr, imag: iPtr, powerSpectrum: pPtr, halfSize: 256)
                    
                    var k = 0
                    while k <= 256 {
                        XCTAssertLessThan(abs(rPtr[k] - 1.0), 1e-4)
                        XCTAssertLessThan(abs(iPtr[k]), 1e-4)
                        XCTAssertLessThan(abs(pPtr[k] - 1.0), 1e-4)
                        k += 1
                    }
                }
            }
        }
        
        // Alternating (-1)^n: all energy (amplitude 512) in bin 256.
        kInit = 0
        while kInit < 512 {
            if (kInit % 2) == 0 {
                real[kInit] = 1.0
            }
            if (kInit % 2) != 0 {
                real[kInit] = -1.0
            }
            imag[kInit] = 0.0
            kInit += 1
        }
        
        real.withUnsafeMutableBufferPointer { rBuf in
            imag.withUnsafeMutableBufferPointer { iBuf in
                power.withUnsafeMutableBufferPointer { pBuf in
                    let rPtr = rBuf.baseAddress!
                    let iPtr = iBuf.baseAddress!
                    let pPtr = pBuf.baseAddress!
                    
                    fft.forward(real: rPtr, imag: iPtr)
                    fft.computePowerSpectrum(real: rPtr, imag: iPtr, powerSpectrum: pPtr, halfSize: 256)
                    
                    XCTAssertLessThan(abs(rPtr[256] - 512.0), 1e-3)
                    XCTAssertLessThan(abs(pPtr[256] - (512.0 * 512.0)), 1.0)
                    
                    var k = 0
                    while k < 256 {
                        XCTAssertLessThan(abs(rPtr[k]), 1e-3)
                        XCTAssertLessThan(abs(iPtr[k]), 1e-3)
                        XCTAssertLessThan(abs(pPtr[k]), 1e-3)
                        k += 1
                    }
                }
            }
        }
        
        // Constant 1.0: amplitude 512 in the DC bin.
        kInit = 0
        while kInit < 512 {
            real[kInit] = 1.0
            imag[kInit] = 0.0
            kInit += 1
        }
        
        real.withUnsafeMutableBufferPointer { rBuf in
            imag.withUnsafeMutableBufferPointer { iBuf in
                power.withUnsafeMutableBufferPointer { pBuf in
                    let rPtr = rBuf.baseAddress!
                    let iPtr = iBuf.baseAddress!
                    let pPtr = pBuf.baseAddress!
                    
                    fft.forward(real: rPtr, imag: iPtr)
                    fft.computePowerSpectrum(real: rPtr, imag: iPtr, powerSpectrum: pPtr, halfSize: 256)
                    
                    XCTAssertLessThan(abs(rPtr[0] - 512.0), 1e-3)
                    XCTAssertLessThan(abs(pPtr[0] - (512.0 * 512.0)), 1.0)
                    
                    var k = 1
                    while k <= 256 {
                        XCTAssertLessThan(abs(rPtr[k]), 1e-3)
                        XCTAssertLessThan(abs(iPtr[k]), 1e-3)
                        XCTAssertLessThan(abs(pPtr[k]), 1e-3)
                        k += 1
                    }
                }
            }
        }
        
        // Subnormal inputs.
        let subnormal = Float.leastNonzeroMagnitude
        kInit = 0
        while kInit < 512 {
            real[kInit] = subnormal
            imag[kInit] = 0.0
            kInit += 1
        }
        
        real.withUnsafeMutableBufferPointer { rBuf in
            imag.withUnsafeMutableBufferPointer { iBuf in
                power.withUnsafeMutableBufferPointer { pBuf in
                    let rPtr = rBuf.baseAddress!
                    let iPtr = iBuf.baseAddress!
                    let pPtr = pBuf.baseAddress!
                    
                    fft.forward(real: rPtr, imag: iPtr)
                    fft.computePowerSpectrum(real: rPtr, imag: iPtr, powerSpectrum: pPtr, halfSize: 256)
                    
                    var k = 0
                    while k <= 256 {
                        XCTAssertFalse(rPtr[k].isNaN)
                        XCTAssertFalse(rPtr[k].isInfinite)
                        XCTAssertFalse(pPtr[k].isNaN)
                        XCTAssertFalse(pPtr[k].isInfinite)
                        k += 1
                    }
                }
            }
        }
    }
    
    /// Parseval's theorem: sum |x[n]|^2 == (1/N) sum |X[k]|^2.
    func testFFTParsevalEnergyConservation() {
        let fft = FFT(size: 512)
        var real = [Float](repeating: 0.0, count: 512)
        var imag = [Float](repeating: 0.0, count: 512)
        
        var n = 0
        var timeDomainEnergy: Float = 0.0
        while n < 512 {
            let t = Float(n) / 512.0
            let val = (0.6 * sin(2.0 * Float.pi * 7.0 * t)) + (0.4 * cos(2.0 * Float.pi * 23.0 * t))
            real[n] = val
            imag[n] = 0.0
            timeDomainEnergy += val * val
            n += 1
        }
        
        real.withUnsafeMutableBufferPointer { rBuf in
            imag.withUnsafeMutableBufferPointer { iBuf in
                let rPtr = rBuf.baseAddress!
                let iPtr = iBuf.baseAddress!
                
                fft.forward(real: rPtr, imag: iPtr)
                
                var freqDomainEnergy: Float = 0.0
                var k = 0
                while k < 512 {
                    let r = rPtr[k]
                    let im = iPtr[k]
                    freqDomainEnergy += (r * r) + (im * im)
                    k += 1
                }
                
                let scaledFreqEnergy = freqDomainEnergy / 512.0
                XCTAssertLessThan(abs(timeDomainEnergy - scaledFreqEnergy), 1e-2)
            }
        }
    }
    
    // MARK: - Mel filterbank range
    
    /// Mel filterbank output stays within [0, 1] for DC, Nyquist, and swept tones.
    func testMelFilterbankBoundaryFrequenciesAndNormalization() {
        let filterbank = Filterbank()
        let workspace = DSPWorkspace()
        let frameSize = 512
        let dummyPitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let dummyFormants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)
        
        let testFrequencies: [Float] = [0.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 2500.0, 4000.0, 7000.0, 7999.0, 8000.0]
        
        for freq in testFrequencies {
            var signal = [Float](repeating: 0.0, count: frameSize)
            var i = 0
            while i < frameSize {
                if freq == 0.0 {
                    signal[i] = 1.0 // DC
                }
                if freq != 0.0 {
                    signal[i] = 0.8 * sin(2.0 * Float.pi * freq * Float(i) / 16000.0)
                }
                i += 1
            }
            
            let features = signal.withUnsafeBufferPointer { ptr in
                filterbank.extractFeatures(
                    pcmPtr: ptr.baseAddress!,
                    count: frameSize,
                    workspace: workspace
                )
            }
            
            XCTAssertEqual(features.count, 64)
            var ch = 0
            while ch < 64 {
                let f = features[ch]
                XCTAssertFalse(f.isNaN, "Freq \(freq)Hz ch \(ch) returned NaN")
                XCTAssertFalse(f.isInfinite, "Freq \(freq)Hz ch \(ch) returned Inf")
                XCTAssertLessThanOrEqual(0.0, f, "Freq \(freq)Hz ch \(ch) was below 0.0 (\(f))")
                XCTAssertLessThanOrEqual(f, 1.0, "Freq \(freq)Hz ch \(ch) exceeded 1.0 (\(f))")
                ch += 1
            }
        }
    }
    
    /// Mel filterbank stays stable on an impulse and a harmonic-rich sawtooth.
    func testMelFilterbankHarmonicsAndPulseStability() {
        let filterbank = Filterbank()
        let workspace = DSPWorkspace()
        let frameSize = 512
        
        // Impulse at n = 0.
        var pulse = [Float](repeating: 0.0, count: frameSize)
        pulse[0] = 1.0
        
        let featPulse = pulse.withUnsafeBufferPointer { ptr in
            filterbank.extractFeatures(
                pcmPtr: ptr.baseAddress!,
                count: frameSize,
                workspace: workspace
            )
        }
        XCTAssertEqual(featPulse.count, 64)
        for f in featPulse {
            XCTAssertFalse(f.isNaN)
            XCTAssertLessThanOrEqual(0.0, f)
            XCTAssertLessThanOrEqual(f, 1.0)
        }
        
        // 200 Hz sawtooth (harmonics decay as 1/k).
        var sawtooth = [Float](repeating: 0.0, count: frameSize)
        var i = 0
        let period = 16000.0 / 200.0 // 80 samples
        while i < frameSize {
            let phase = Float(i).truncatingRemainder(dividingBy: Float(period))
            sawtooth[i] = (2.0 * (phase / Float(period))) - 1.0
            i += 1
        }
        
        let featSaw = sawtooth.withUnsafeBufferPointer { ptr in
            filterbank.extractFeatures(
                pcmPtr: ptr.baseAddress!,
                count: frameSize,
                workspace: workspace
            )
        }
        XCTAssertEqual(featSaw.count, 64)
        for f in featSaw {
            XCTAssertFalse(f.isNaN)
            XCTAssertLessThanOrEqual(0.0, f)
            XCTAssertLessThanOrEqual(f, 1.0)
        }
    }
    
    /// Output is clamped to [0, 1] for amplitudes of 1e10 and 1e-15.
    func testMelFilterbankExtremeAmplitudesRangeCheck() {
        let filterbank = Filterbank()
        let workspace = DSPWorkspace()
        let frameSize = 512
        
        let scales: [Float] = [1e10, 1e5, 1e-5, 1e-15]
        
        for scale in scales {
            var signal = [Float](repeating: 0.0, count: frameSize)
            var i = 0
            while i < frameSize {
                signal[i] = scale * sin(2.0 * Float.pi * 440.0 * Float(i) / 16000.0)
                i += 1
            }
            
            let features = signal.withUnsafeBufferPointer { ptr in
                filterbank.extractFeatures(
                    pcmPtr: ptr.baseAddress!,
                    count: frameSize,
                    workspace: workspace
                )
            }
            
            XCTAssertEqual(features.count, 64)
            var ch = 0
            while ch < 64 {
                let f = features[ch]
                XCTAssertFalse(f.isNaN, "Scale \(scale) ch \(ch) returned NaN")
                XCTAssertFalse(f.isInfinite, "Scale \(scale) ch \(ch) returned Inf")
                XCTAssertLessThanOrEqual(0.0, f, "Scale \(scale) ch \(ch) was below 0.0 (\(f))")
                XCTAssertLessThanOrEqual(f, 1.0, "Scale \(scale) ch \(ch) exceeded 1.0 (\(f))")
                ch += 1
            }
        }
    }
}

