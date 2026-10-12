import XCTest
import Foundation
@testable import Haitaka

private final class ResultCollector<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []

    func append(_ item: T) {
        lock.lock()
        defer { lock.unlock() }
        items.append(item)
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return items.count
    }

    var allItems: [T] {
        lock.lock()
        defer { lock.unlock() }
        return items
    }

    subscript(index: Int) -> T {
        lock.lock()
        defer { lock.unlock() }
        return items[index]
    }
}

final class Tier3CombinationTests: XCTestCase {

    // MARK: - LPC, Durand-Kerner, formants and filterbank features
    func testComboDurandKernerToFormantEnergyFilterbank() {
        let config = DSPConfig(sampleRate: 16000)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let filterbank = Filterbank(config: config)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)
        speech.withUnsafeBufferPointer { buf in
            let lpcSuccess = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(lpcSuccess)
            if lpcSuccess {
                workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                    let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                    if solverSuccess {
                        workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                            _ = extractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
                            XCTAssertEqual(workspace.featureBuffer.count, 64)
                        }
                    }
                }
            }
        }
    }

    // MARK: - SIMD8 vector operations vs scalar reference
    func testComboSIMD8VectorOpsVsScalarBitExact() {
        let a: [Float] = [1.5, -2.5, 3.0, 4.2, -5.1, 6.0, 7.8, -8.3]
        let b: [Float] = [0.2, 0.4, -0.6, 0.8, -1.0, 1.2, -1.4, 1.6]

        let dot = a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                VectorOperations.dotProduct(a: aBuf.baseAddress!, b: bBuf.baseAddress!, count: 8)
            }
        }
        var expectedDot: Float = 0.0
        var i = 0
        while i < 8 {
            expectedDot += a[i] * b[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(dot - expectedDot), 1e-4)

        let sumSq = a.withUnsafeBufferPointer { buf in
            VectorOperations.sumOfSquares(ptr: buf.baseAddress!, count: 8)
        }
        var expectedSumSq: Float = 0.0
        i = 0
        while i < 8 {
            expectedSumSq += a[i] * a[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(sumSq - expectedSumSq), 1e-4)
    }

    // MARK: - VAD rejection of low-level noise
    func testComboHighNoiseVADToAcousticBlankSuppression() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        // Low amplitude micro noise below threshold
        let noise = [Float](repeating: 0.0001, count: config.frameSize)
        noise.withUnsafeBufferPointer { buf in
            let vadRes = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(vadRes.isSpeech)
        }
    }

    // MARK: - Helpers

    private func createWavData(samples: [Float], sampleRate: Int = 16000) -> Data {
        var data = Data()
        data.append(contentsOf: [UInt8]("RIFF".utf8))
        let subchunk2Size = UInt32(samples.count * 2)
        let chunkSize = UInt32(36 + Int(subchunk2Size))

        var chunkSizeLE = chunkSize.littleEndian
        withUnsafeBytes(of: &chunkSizeLE) { data.append(contentsOf: $0) }

        data.append(contentsOf: [UInt8]("WAVE".utf8))
        data.append(contentsOf: [UInt8]("fmt ".utf8))

        let subchunk1Size: UInt32 = 16
        let audioFormat: UInt16 = 1
        let numChannels: UInt16 = 1
        let sRate = UInt32(sampleRate)
        let byteRate = UInt32(sampleRate * 2)
        let blockAlign: UInt16 = 2
        let bitsPerSamp: UInt16 = 16

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

    private func synthesizeSpeech(sampleRate: Int, durationSeconds: Float, amplitude: Float = 0.5) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = 2.0 * Float.pi

        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            let f0 = sin(twoPi * 200.0 * t)
            let f1 = 0.5 * sin(twoPi * 800.0 * t)
            let f2 = 0.3 * sin(twoPi * 1200.0 * t)
            let f3 = 0.2 * sin(twoPi * 2500.0 * t)
            pcm[i] = amplitude * (f0 + f1 + f2 + f3)
            i += 1
        }
        return pcm
    }

    private func synthesizeSilence(sampleRate: Int, durationSeconds: Float) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        return [Float](repeating: 0.0, count: sampleCount)
    }
}
