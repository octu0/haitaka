import XCTest
import Foundation
@testable import Haitaka
#if canImport(Darwin)
import Darwin
#endif

final class Tier5AdversarialTests: XCTestCase {

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

    private func generateGaussianNoise(count: Int, rms: Float, seed: UInt64 = 42) -> [Float] {
        var result = [Float](repeating: 0.0, count: count)
        var state = seed
        func lcg() -> Float {
            state = (state &* 6364136223846793005 &+ 1442695040888963407)
            let u = Float(Double(state >> 32) / 4294967296.0)
            return max(1e-7, min(0.999999, u))
        }
        var i = 0
        while i < count {
            let u1 = lcg()
            let u2 = lcg()
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            let z0 = r * cos(theta)
            let z1 = r * sin(theta)
            result[i] = z0 * rms
            if (i + 1) < count {
                result[i + 1] = z1 * rms
            }
            i += 2
        }
        return result
    }

    private func generatePinkNoise(count: Int, rms: Float, seed: UInt64 = 42) -> [Float] {
        var result = [Float](repeating: 0.0, count: count)
        var b0: Float = 0.0
        var b1: Float = 0.0
        var b2: Float = 0.0
        var b3: Float = 0.0
        var b4: Float = 0.0
        var b5: Float = 0.0
        var b6: Float = 0.0
        let white = generateGaussianNoise(count: count, rms: 1.0, seed: seed)
        var i = 0
        while i < count {
            let w = white[i]
            b0 = (0.99886 * b0) + (w * 0.0555179)
            b1 = (0.99332 * b1) + (w * 0.0750759)
            b2 = (0.96900 * b2) + (w * 0.1538520)
            b3 = (0.86650 * b3) + (w * 0.3104856)
            b4 = (0.55000 * b4) + (w * 0.5329522)
            b5 = (-0.7616 * b5) - (w * 0.0168980)
            let pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + (w * 0.5362)
            b6 = w * 0.115926
            result[i] = pink * 0.11 * rms
            i += 1
        }
        return result
    }

    // MARK: - Corrupted WAV headers and invalid chunk sizes
    func testAdversarialCorruptedWavByteStreams() {
        let parser = WavParser()
        // Random junk bytes
        let junkData = [UInt8](Data([0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04]))
        XCTAssertThrowsError(try parser.parse(bytes: junkData))

        // Incomplete RIFF header
        let riffData = [UInt8]("RIFF".utf8)
        XCTAssertThrowsError(try parser.parse(bytes: riffData))
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
}
