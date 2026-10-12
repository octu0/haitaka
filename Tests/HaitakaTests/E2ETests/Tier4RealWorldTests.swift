import XCTest
import Foundation
@testable import Haitaka
#if canImport(Darwin)
import Darwin
#endif

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

final class Tier4RealWorldTests: XCTestCase {

    private func getResidentMemoryBytes() -> UInt64 {
        #if canImport(Darwin)
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return taskInfo.resident_size
        }
        return 0
        #else
        return 0
        #endif
    }

    // MARK: - Five Japanese vowels (a, i, u, e, o) in sequence
    func testScenario1JapaneseVowelsAIEUO() {
        let sampleRate = 16000
        let frameSize = 512
        let config = DSPConfig(sampleRate: sampleRate, frameSize: frameSize, maxPitchLag: 320)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels, maxPitchLag: 320)
        let vad = VAD(config: config)
        let pitchDetector = PitchDetector(config: config)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let formantExtractor = FormantExtractor(sampleRate: Float(sampleRate))
        let filterbank = Filterbank(config: config)

        struct VowelSpec {
            let name: String
            let f1: Float
            let f2: Float
            let f3: Float
        }
        let vowels: [VowelSpec] = [
            VowelSpec(name: "a", f1: 800.0, f2: 1300.0, f3: 2600.0),
            VowelSpec(name: "i", f1: 300.0, f2: 2300.0, f3: 3000.0),
            VowelSpec(name: "u", f1: 380.0, f2: 1200.0, f3: 2400.0),
            VowelSpec(name: "e", f1: 500.0, f2: 1900.0, f3: 2600.0),
            VowelSpec(name: "o", f1: 500.0, f2: 900.0, f3: 2500.0)
        ]

        var vIdx = 0
        while vIdx < vowels.count {
            let v = vowels[vIdx]
            var wave = [Float](repeating: 0.0, count: frameSize)
            let twoPi = 2.0 * Float.pi
            var i = 0
            while i < frameSize {
                let t = Float(i) / Float(sampleRate)
                let f0 = 0.5 * sin(twoPi * 150.0 * t)
                let f1 = 0.4 * sin(twoPi * v.f1 * t)
                let f2 = 0.3 * sin(twoPi * v.f2 * t)
                let f3 = 0.2 * sin(twoPi * v.f3 * t)
                wave[i] = f0 + f1 + f2 + f3
                i += 1
            }

            wave.withUnsafeBufferPointer { buf in
                let framePtr = buf.baseAddress!
                let vadRes = vad.processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertTrue(vadRes.isSpeech)

                let pitchRes = pitchDetector.detectPitch(ptr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertTrue(pitchRes.isVoiced)

                let lpcSuccess = lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertTrue(lpcSuccess)

                var formantRes = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)
                if lpcSuccess {
                    workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                        let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                        if solverSuccess {
                            workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                                formantRes = formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                            }
                        }
                    }
                }

                filterbank.extractFeatures(pcmPtr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertEqual(workspace.featureBuffer.count, 64)
            }

            vIdx += 1
        }
    }

    // MARK: - High and low pitch
    func testScenario6HighPitchAndLowPitchAdaptation() {
        let sampleRate = 16000
        let frameSize = 512
        let config = DSPConfig(sampleRate: sampleRate, frameSize: frameSize, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels, maxPitchLag: 320)

        // Male low pitch (100Hz)
        var maleVoice = [Float](repeating: 0.0, count: frameSize)
        // Female or child high pitch (320Hz)
        var femaleVoice = [Float](repeating: 0.0, count: frameSize)

        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < frameSize {
            let t = Float(i) / Float(sampleRate)
            maleVoice[i] = 0.6 * sin(twoPi * 100.0 * t) + 0.3 * sin(twoPi * 200.0 * t)
            femaleVoice[i] = 0.6 * sin(twoPi * 320.0 * t) + 0.3 * sin(twoPi * 640.0 * t)
            i += 1
        }

        maleVoice.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(80.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 150.0)
        }

        femaleVoice.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(280.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 360.0)
        }
    }

    // MARK: - Helpers

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
