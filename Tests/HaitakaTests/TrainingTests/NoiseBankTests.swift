import XCTest
@testable import Haitaka

final class NoiseBankTests: XCTestCase {
    private func tone(count: Int, amplitude: Float, period: Int) -> [Float] {
        var out = [Float](repeating: 0.0, count: count)
        var i = 0
        while i < count {
            out[i] = amplitude * sin(Float(i) * 2.0 * Float.pi / Float(period))
            i += 1
        }
        return out
    }

    func testMixMatchesRequestedSNR() {
        let speech = tone(count: 16000, amplitude: 0.2, period: 100)
        let noise = tone(count: 4000, amplitude: 0.7, period: 37)
        let mixed = NoiseBank.mix(speech: speech, noise: noise, offset: 123, snrDB: 10.0)
        XCTAssertEqual(mixed.count, speech.count)
        var residual = [Float](repeating: 0.0, count: speech.count)
        var i = 0
        while i < speech.count {
            residual[i] = mixed[i] - speech[i]
            i += 1
        }
        let snr = 20.0 * log10(NoiseBank.rms(speech) / NoiseBank.rms(residual))
        XCTAssertEqual(snr, 10.0, accuracy: 0.2)
    }

    func testShortNoiseIsTiledFromOffset() {
        let speech = [Float](repeating: 0.1, count: 10)
        let noise: [Float] = [1.0, 2.0, 3.0]
        let mixed = NoiseBank.mix(speech: speech, noise: noise, offset: 1, snrDB: 0.0)
        // Noise tiles as 2,3,1,2,3,...; at 0 dB SNR its RMS equals the speech RMS (0.1).
        let residual0 = mixed[0] - speech[0]
        let residual1 = mixed[1] - speech[1]
        let residual2 = mixed[2] - speech[2]
        XCTAssertEqual(residual1 / residual0, 1.5, accuracy: 1e-4)
        XCTAssertEqual(residual2 / residual0, 0.5, accuracy: 1e-4)
        var acc: Float = 0.0
        var i = 0
        while i < mixed.count {
            let r = mixed[i] - speech[i]
            acc += r * r
            i += 1
        }
        XCTAssertEqual((acc / Float(mixed.count)).squareRoot(), 0.1, accuracy: 1e-3)
    }

    func testSilentSpeechAndEmptyNoiseAreLeftUntouched() {
        let silence = [Float](repeating: 0.0, count: 100)
        XCTAssertEqual(NoiseBank.mix(speech: silence, noise: [0.5, -0.5], offset: 0, snrDB: 5.0), silence)
        let speech = tone(count: 100, amplitude: 0.3, period: 10)
        XCTAssertEqual(NoiseBank.mix(speech: speech, noise: [], offset: 0, snrDB: 5.0), speech)
    }

    /// 0.6 s of tone followed by 0.4 s of silence: the voiced frames are the tone.
    private func toneThenSilence(amplitude: Float) -> [Float] {
        return tone(count: 9600, amplitude: amplitude, period: 100) + [Float](repeating: 0.0, count: 6400)
    }

    private func residualPower(_ mixed: [Float], _ speech: [Float]) -> Float {
        var acc: Float = 0.0
        var i = 0
        while i < mixed.count {
            let r = mixed[i] - speech[i]
            acc += r * r
            i += 1
        }
        return acc / Float(mixed.count)
    }

    func testLevelsMeasureVoicedFramesAndFloor() {
        let speech = toneThenSilence(amplitude: 0.2)
        let level = NoiseBank.levels(speech)
        XCTAssertEqual(level.voiced, 0.02, accuracy: 1e-3)
        XCTAssertEqual(level.floor, 0.0, accuracy: 1e-9)
    }

    func testBankReachesTargetSNROverVoicedFrames() {
        let bank = NoiseBank(clips: [tone(count: 8000, amplitude: 0.5, period: 41), []])
        XCTAssertEqual(bank.count, 1)
        XCTAssertEqual(bank.totalSeconds, 0.5, accuracy: 1e-6)
        let speech = toneThenSilence(amplitude: 0.2)
        let voiced = NoiseBank.levels(speech).voiced
        var rng = SplitMix64(seed: 5)
        var mixedCount = 0
        var trial = 0
        while trial < 40 {
            let noise = residualPower(bank.mix(speech, using: &rng), speech)
            if 1e-12 < noise {
                mixedCount += 1
                // Pauses do not lower the reference: the SNR is against the tone, not the whole-utterance RMS.
                let snr = 10.0 * log10(voiced / noise)
                XCTAssertGreaterThanOrEqual(snr, bank.snrRange.lowerBound - 0.3)
                XCTAssertLessThanOrEqual(snr, bank.snrRange.upperBound + 0.3)
            }
            trial += 1
        }
        // A clean recording is mixed with probability 0.7, so some of the 40 trials pass through unchanged.
        XCTAssertGreaterThan(mixedCount, 15)
        XCTAssertLessThan(mixedCount, 38)
    }

    func testNoiseAlreadyPresentCountsTowardTheTarget() {
        var noiseRNG = SplitMix64(seed: 11)
        var speech = toneThenSilence(amplitude: 0.2)
        var floor = [Float](repeating: 0.0, count: speech.count)
        var i = 0
        while i < speech.count {
            // Uniform noise of RMS 0.02: 0.0004 against the tone's 0.02, a 17 dB floor already in the recording.
            floor[i] = Float.random(in: -0.0346...0.0346, using: &noiseRNG)
            speech[i] += floor[i]
            i += 1
        }
        let clip = tone(count: 8000, amplitude: 0.5, period: 41)
        var rng = SplitMix64(seed: 2)
        let quieter = NoiseBank(clips: [clip], probability: 1.0, recordedProbability: 1.0, snrRange: 25.0...25.0)
        XCTAssertEqual(quieter.mix(speech, using: &rng), speech)
        let louder = NoiseBank(clips: [clip], probability: 1.0, recordedProbability: 1.0, snrRange: 10.0...10.0)
        let added = residualPower(louder.mix(speech, using: &rng), speech)
        let level = NoiseBank.levels(speech)
        // Only the missing part is added: about 0.02 / 10 - 0.0004 = 0.0016, not the full 0.002.
        XCTAssertEqual(added, level.voiced / 10.0 - level.floor, accuracy: 2e-4)
        XCTAssertLessThan(added, level.voiced / 10.0 * 0.9)
        // A recording with its own background sound is mixed with `recordedProbability`, a clean one with `probability`.
        let recordedOnly = NoiseBank(clips: [clip], probability: 0.0, recordedProbability: 1.0, snrRange: 10.0...10.0)
        XCTAssertNotEqual(recordedOnly.mix(speech, using: &rng), speech)
        let clean = toneThenSilence(amplitude: 0.2)
        XCTAssertEqual(recordedOnly.mix(clean, using: &rng), clean)
    }

    func testOverlayAddsVoiceOnceAtItsOwnLevel() {
        let speech = [Float](repeating: 0.1, count: 10)
        let voice: [Float] = [1.0, -1.0, 1.0, -1.0]
        // At 0 dB the voice's own RMS (1.0) is scaled to the speech RMS (0.1).
        let mixed = NoiseBank.overlay(speech: speech, voice: voice, start: 3, snrDB: 0.0)
        let expected: [Float] = [0.1, 0.1, 0.1, 0.2, 0.0, 0.2, 0.0, 0.1, 0.1, 0.1]
        var i = 0
        while i < expected.count {
            XCTAssertEqual(mixed[i], expected[i], accuracy: 1e-5)
            i += 1
        }
    }

    func testOverlayCutsVoiceAtBothEnds() {
        let speech = [Float](repeating: 0.1, count: 4)
        let voice: [Float] = [1.0, -1.0, 1.0, -1.0]
        let head = NoiseBank.overlay(speech: speech, voice: voice, start: -3, snrDB: 0.0)
        XCTAssertEqual(head[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(head[1], 0.1, accuracy: 1e-5)
        let tail = NoiseBank.overlay(speech: speech, voice: voice, start: 3, snrDB: 0.0)
        XCTAssertEqual(tail[2], 0.1, accuracy: 1e-5)
        XCTAssertEqual(tail[3], 0.2, accuracy: 1e-5)
    }

    func testVoiceBankUsesVoiceSettings() {
        let bank = NoiseBank.voice(clips: [tone(count: 4000, amplitude: 0.5, period: 41)])
        XCTAssertEqual(bank.probability, NoiseBank.voiceProbability)
        XCTAssertEqual(bank.snrRange, NoiseBank.voiceSNRRange)
        XCTAssertEqual(bank.placement, .once)
        let speech = tone(count: 16000, amplitude: 0.2, period: 100)
        var rng = SystemRandomNumberGenerator()
        var trial = 0
        while trial < 20 {
            XCTAssertEqual(bank.mix(speech, using: &rng).count, speech.count)
            trial += 1
        }
    }
}
