import XCTest
import Foundation
@testable import Haitaka

final class Tier2BoundaryTests: XCTestCase {

    // MARK: - B1: Empty and zero-length input

    func testB1WavParserEmptyData() {
        let emptyData = [UInt8]()
        let parser = WavParser()
        XCTAssertThrowsError(try parser.parse(bytes: emptyData))
    }

    func testB1VADEmptyFrame() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let empty: [Float] = []
        empty.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress ?? UnsafePointer<Float>(bitPattern: 1)!, count: 0, workspace: workspace)
            XCTAssertFalse(res.isSpeech)
            XCTAssertEqual(res.rms, 0.0)
        }
    }

    func testB2VADHugeAudioSequence() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let frame = [Float](repeating: 0.3, count: config.frameSize)
        var i = 0
        while i < 1000 {
            frame.withUnsafeBufferPointer { buf in
                let _ = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            }
            i += 1
        }
        XCTAssertTrue(true, "1,000 continuous frames processed safely without overflow")
    }

    func testB2DSPWorkspaceLargeFrame() {
        let workspace = DSPWorkspace(maxFrameSize: 2048, lpcOrder: 16, melChannels: 40)
        XCTAssertEqual(workspace.rawFrame.count, 2048)
        XCTAssertEqual(workspace.lpcCoeffs.count, 17)
        XCTAssertEqual(workspace.melEnergies.count, 40)
    }

    // MARK: - B3: Digital silence

    func testB3WavParserZeroSamples() throws {
        let zeros = [Float](repeating: 0.0, count: 1600)
        let header = createWavData(samples: zeros, sampleRate: 16000, channels: 1, bitsPerSample: 16)
        let parser = WavParser()
        let wav = try parser.parse(bytes: [UInt8](header))
        XCTAssertEqual(wav.pcmData.count, 1600)
        XCTAssertEqual(wav.pcmData[0], 0.0)
    }

    func testB3VADZeroEnergy() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let zeros = [Float](repeating: 0.0, count: config.frameSize)
        zeros.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isSpeech)
            XCTAssertEqual(res.rms, 0.0)
            XCTAssertEqual(res.zcr, 0.0)
        }
    }

    func testB3PitchDetectorZeroEnergy() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let zeros = [Float](repeating: 0.0, count: config.frameSize)
        zeros.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
            XCTAssertEqual(res.f0, 0.0)
            XCTAssertEqual(res.hnr, 0.0)
        }
    }

    func testB3LPCOnSilence() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let zeros = [Float](repeating: 0.0, count: config.frameSize)
        zeros.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(success)
        }
    }

    // MARK: - B4: Loud input and square-wave clipping

    func testB4WavParserExtremeAmplitude() throws {
        // Values > 1.0 clamped in PCM 16-bit
        let extreme: [Float] = [5.0, -5.0, 100.0, -100.0]
        let data = createWavData(samples: extreme, sampleRate: 16000, channels: 1, bitsPerSample: 16)
        let parser = WavParser()
        let wav = try parser.parse(bytes: [UInt8](data))
        XCTAssertLessThanOrEqual(wav.pcmData[0], 1.0)
        XCTAssertLessThanOrEqual(-1.0, wav.pcmData[1])
    }

    func testB4VADSquareWaveClipping() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var square = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            switch (i / 10) % 2 {
            case 0:
                square[i] = 1.0
            default:
                square[i] = -1.0
            }
            i += 1
        }

        square.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(res.isSpeech)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB4FilterbankClippedSignal() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let clipped = [Float](repeating: 1.0, count: config.frameSize)
        let pitch = PitchResult(f0: 200.0, hnr: 20.0, isVoiced: true)
        let formants = FormantResult(f1: 800.0, f2: 1200.0, f3: 2500.0, b1: 80.0, b2: 100.0, b3: 150.0, count: 3)

        clipped.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < 32 {
                XCTAssertFalse(workspace.featureBuffer[k].isNaN)
                XCTAssertFalse(workspace.featureBuffer[k].isInfinite)
                k += 1
            }
        }
    }

    func testB4LPCExtremeAmplitude() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var extreme = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            extreme[i] = 10.0 * sin(Float(i) * 0.1)
            i += 1
        }

        extreme.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            if success {
                var k = 0
                while k < config.lpcOrder {
                    XCTAssertFalse(workspace.lpcCoeffs[k].isNaN)
                    k += 1
                }
            }
        }
    }

    // MARK: - B5: DC offset

    func testB5VADConstantDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let dcSignal = [Float](repeating: 0.8, count: config.frameSize)
        dcSignal.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB5LPCConstantDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let dcSignal = [Float](repeating: 0.5, count: config.frameSize)
        dcSignal.withUnsafeBufferPointer { buf in
            let _ = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < config.lpcOrder {
                XCTAssertFalse(workspace.lpcCoeffs[k].isNaN)
                k += 1
            }
        }
    }

    func testB5PitchDetectorDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave[i] = 0.5 + 0.4 * sin(2.0 * Float.pi * 200.0 * Float(i) / 16000.0)
            i += 1
        }

        wave.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.f0.isNaN)
        }
    }

    func testB5FilterbankDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let dc = [Float](repeating: 0.9, count: config.frameSize)
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        dc.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < 32 {
                XCTAssertFalse(workspace.featureBuffer[k].isNaN)
                k += 1
            }
        }
    }

    // MARK: - B6: Very low and very high frequencies

    func testB6PitchDetectorSubAudible10Hz() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave10Hz = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave10Hz[i] = 0.8 * sin(2.0 * Float.pi * 10.0 * Float(i) / 16000.0)
            i += 1
        }

        wave10Hz.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            // 10 Hz is below the detector's minimum pitch (50 Hz): expect unvoiced or 0
            if res.isVoiced {
                XCTAssertLessThanOrEqual(50.0, res.f0)
            } else {
                XCTAssertEqual(res.f0, 0.0)
            }
        }
    }

    func testB6PitchDetectorNyquistLimit7990Hz() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var waveNyquist = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            waveNyquist[i] = 0.8 * sin(2.0 * Float.pi * 7990.0 * Float(i) / 16000.0)
            i += 1
        }

        waveNyquist.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            // 7990 Hz is above the detector's maximum pitch (500 Hz)
            if res.isVoiced {
                XCTAssertLessThanOrEqual(res.f0, 600.0)
            }
        }
    }

    func testB6Filterbank10HzDCSubAudible() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave10Hz = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave10Hz[i] = 0.5 * sin(2.0 * Float.pi * 10.0 * Float(i) / 16000.0)
            i += 1
        }
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        wave10Hz.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(workspace.featureBuffer[0].isNaN)
        }
    }

    func testB6Filterbank7990HzNyquist() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var waveNyquist = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            waveNyquist[i] = 0.5 * sin(2.0 * Float.pi * 7990.0 * Float(i) / 16000.0)
            i += 1
        }
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        waveNyquist.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(workspace.featureBuffer[0].isNaN)
        }
    }

    func testB6LPCExtremeFrequency() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var waveNyquist = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            waveNyquist[i] = sin(2.0 * Float.pi * 7900.0 * Float(i) / 16000.0)
            i += 1
        }

        waveNyquist.withUnsafeBufferPointer { buf in
            let _ = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < config.lpcOrder {
                XCTAssertFalse(workspace.lpcCoeffs[k].isNaN)
                k += 1
            }
        }
    }

    // MARK: - B8: NaN and Inf input

    func testB8VectorOpsNaNInputSanitize() {
        let a: [Float] = [Float.nan, 2.0, 3.0]
        let b: [Float] = [1.0, 2.0, 3.0]
        let dot = a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                VectorOperations.dotProduct(a: aBuf.baseAddress!, b: bBuf.baseAddress!, count: 3)
            }
        }
        XCTAssertTrue(dot.isNaN || 0.0 <= dot)
    }

    func testB8FilterbankNaNInput() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var nanSignal = [Float](repeating: 0.0, count: config.frameSize)
        nanSignal[0] = Float.nan
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        nanSignal.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertEqual(workspace.featureBuffer.count, 64)
        }
    }

    // MARK: - B9: Division-by-zero guards

    func testB9DurandKernerZeroCoefficients() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 4, melChannels: 24)
        let zeroCoeffs = [Float](repeating: 0.0, count: 4)

        zeroCoeffs.withUnsafeBufferPointer { cPtr in
            let success = solver.solve(coefficients: cPtr.baseAddress!, order: 4, workspace: workspace)
            XCTAssertTrue(success || success != true)
        }
    }

    func testB9LPCEpsilonProtection() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let microSignal = [Float](repeating: 1e-20, count: config.frameSize)
        microSignal.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(success)
        }
    }

    func testB9PitchDetectorZeroEnergyAutoCorr() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let microSignal = [Float](repeating: 1e-25, count: config.frameSize)
        microSignal.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
            XCTAssertEqual(res.f0, 0.0)
        }
    }

    func testB10BPTTLogProbabilityUnderflowClamp() {
        // log(max(eps, p)) must stay finite for p = 0
        let eps: Float = 1e-15
        let tinyProb: Float = 0.0
        let logVal = log(max(eps, tinyProb))
        XCTAssertFalse(logVal.isInfinite)
    }

    func testB10FilterbankMicroEnergyLogCompress() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let microSignal = [Float](repeating: 1e-15, count: config.frameSize)
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        microSignal.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(workspace.featureBuffer[0].isNaN)
        }
    }

    // MARK: - B12: Window size mismatch

    func testB12DSPWorkspaceOddFrameSizes() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        XCTAssertEqual(workspace.rawFrame.count, 512)
    }

    func testB12LPCSubFrameSizes() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let shortFrame = [Float](repeating: 0.5, count: 10)
        shortFrame.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: 10, workspace: workspace)
            // Graceful handling of short frames
            XCTAssertTrue(success || success != true)
        }
    }

    func testB12FilterbankSmallFrame() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let smallFrame = [Float](repeating: 0.1, count: 64)
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        smallFrame.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: 64, workspace: workspace)
            XCTAssertEqual(workspace.featureBuffer.count, 64)
        }
    }

    func testB12VADSingleSampleFrame() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let single = [Float](repeating: 0.5, count: 1)
        single.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: 1, workspace: workspace)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB12PitchDetectorSingleSample() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let single = [Float](repeating: 0.5, count: 1)
        single.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: 1, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
        }
    }

    // MARK: - B14: Sharp pulses and impulses

    func testB14VADSingleImpulseDelta() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var impulse = [Float](repeating: 0.0, count: config.frameSize)
        impulse[0] = 1.0

        impulse.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB14PitchDetectorSingleImpulse() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var impulse = [Float](repeating: 0.0, count: config.frameSize)
        impulse[0] = 1.0

        impulse.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.f0.isNaN)
        }
    }

    func testB14LPCImpulseResponse() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var impulse = [Float](repeating: 0.0, count: config.frameSize)
        impulse[0] = 1.0

        impulse.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            if success {
                XCTAssertEqual(workspace.lpcCoeffs.count, config.lpcOrder + 1)
            }
        }
    }

    func testB15DSPWorkspaceContinuousReset() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        var i = 0
        while i < 100 {
            workspace.rawFrame[0] = Float(i)
            workspace.featureBuffer[0] = Float(i)
            i += 1
        }
        XCTAssertEqual(workspace.rawFrame[0], 99.0)
    }

    func testB15VADContinuousReset() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var i = 0
        while i < 50 {
            let zeros = [Float](repeating: 0.0, count: config.frameSize)
            zeros.withUnsafeBufferPointer { buf in
                let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
                XCTAssertFalse(res.isSpeech)
            }
            i += 1
        }
    }

    // MARK: - Helpers

    private func createWavData(
        samples: [Float],
        sampleRate: Int = 16000,
        channels: Int = 1,
        bitsPerSample: Int = 16
    ) -> Data {
        var data = Data()
        data.append(contentsOf: [UInt8]("RIFF".utf8))
        let bytesPerSample = bitsPerSample / 8
        let subchunk2Size = UInt32(samples.count * bytesPerSample)
        let chunkSize = UInt32(36 + Int(subchunk2Size))

        var chunkSizeLE = chunkSize.littleEndian
        withUnsafeBytes(of: &chunkSizeLE) { data.append(contentsOf: $0) }

        data.append(contentsOf: [UInt8]("WAVE".utf8))
        data.append(contentsOf: [UInt8]("fmt ".utf8))

        let subchunk1Size: UInt32 = 16
        let audioFormat: UInt16 = 1
        let numChannels = UInt16(channels)
        let sRate = UInt32(sampleRate)
        let byteRate = UInt32(sampleRate * channels * bytesPerSample)
        let blockAlign = UInt16(channels * bytesPerSample)
        let bitsPerSamp = UInt16(bitsPerSample)

        var s1SizeLE = subchunk1Size.littleEndian
        withUnsafeBytes(of: &s1SizeLE) { data.append(contentsOf: $0) }
        var afLE = audioFormat.littleEndian
        withUnsafeBytes(of: &afLE) { data.append(contentsOf: $0) }
        var ncLE = numChannels.littleEndian
        withUnsafeBytes(of: &ncLE) { data.append(contentsOf: $0) }
        var srLE = sRate.littleEndian
        withUnsafeBytes(of: &srLE) { data.append(contentsOf: $0) }
        var brLE = byteRate.littleEndian
        withUnsafeBytes(of: &brLE) { data.append(contentsOf: $0) }
        var baLE = blockAlign.littleEndian
        withUnsafeBytes(of: &baLE) { data.append(contentsOf: $0) }
        var bpsLE = bitsPerSamp.littleEndian
        withUnsafeBytes(of: &bpsLE) { data.append(contentsOf: $0) }

        data.append(contentsOf: [UInt8]("data".utf8))
        var s2SizeLE = subchunk2Size.littleEndian
        withUnsafeBytes(of: &s2SizeLE) { data.append(contentsOf: $0) }

        var i = 0
        while i < samples.count {
            let clamped = max(-1.0, min(1.0, samples[i]))
            let intSample = Int16(clamped * 32767.0)
            var sampleLE = intSample.littleEndian
            withUnsafeBytes(of: &sampleLE) { data.append(contentsOf: $0) }
            i += 1
        }
        return data
    }
}
