import XCTest
@testable import Haitaka

final class SpeechDatasetTests: XCTestCase {
    func testFeaturesHaveModelInputWidth() {
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < pcm.count {
            pcm[i] = 0.1 * sinf(2.0 * Float.pi * 220.0 * Float(i) / 16000.0)
            i += 1
        }
        let features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm)
        XCTAssertEqual(features.count, 24)
        for frame in features {
            XCTAssertEqual(frame.count, StreamingFeatureFrontEnd.inputDim)
        }
    }

    private func tone(_ hz: Float, rate: Int, seconds: Float) -> [Float] {
        let n = Int(Float(rate) * seconds)
        var pcm = [Float](repeating: 0.0, count: n)
        var i = 0
        while i < n {
            pcm[i] = sinf(2.0 * Float.pi * hz * Float(i) / Float(rate))
            i += 1
        }
        return pcm
    }

    /// RMS over the middle half, away from the edges where the kernel is cut.
    private func middleRMS(_ x: [Float]) -> Float {
        var acc: Float = 0.0
        var i = x.count / 4
        while i < x.count * 3 / 4 {
            acc += x[i] * x[i]
            i += 1
        }
        return (acc / Float(x.count / 2)).squareRoot()
    }

    func testResampleKeepsSixteenKilohertzUnchanged() {
        let pcm = tone(440.0, rate: 16000, seconds: 0.1)
        XCTAssertEqual(SpeechDataset.resampleTo16k(pcmData: pcm, sampleRate: 16000), pcm)
    }

    func testResampleKeepsDirectCurrentAndLength() {
        for rate in [48000, 44100, 24000, 96000] {
            let pcm = [Float](repeating: 0.5, count: rate / 10)
            let out = SpeechDataset.resampleTo16k(pcmData: pcm, sampleRate: rate)
            XCTAssertEqual(out.count, Int(Float(pcm.count) / (Float(rate) / 16000.0)))
            for v in out {
                XCTAssertEqual(v, 0.5, accuracy: 1e-4)
            }
        }
    }

    func testResamplePassesSpeechBandAndRemovesAliases() {
        for rate in [48000, 44100, 24000] {
            let pass = SpeechDataset.resampleTo16k(pcmData: tone(1000.0, rate: rate, seconds: 0.5), sampleRate: rate)
            XCTAssertEqual(middleRMS(pass), Float(0.5).squareRoot(), accuracy: 0.01, "1 kHz at \(rate) Hz")
            // 11 kHz is above the 8 kHz output Nyquist frequency; without the low-pass it folds back to 5 kHz.
            let alias = SpeechDataset.resampleTo16k(pcmData: tone(11000.0, rate: rate, seconds: 0.5), sampleRate: rate)
            XCTAssertLessThan(middleRMS(alias), 0.01, "11 kHz at \(rate) Hz")
        }
    }

    /// Appended samples do not change the gain, so the utterance's own frames keep their features.
    func testGainComesFromTheGivenSamples() {
        var pcm = tone(440.0, rate: 16000, seconds: 0.5)
        pcm += [Float](repeating: 0.0, count: 8000)
        let plain = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm)
        let padded = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm + [Float](repeating: 0.0, count: 8000), rmsSampleCount: pcm.count)
        XCTAssertLessThan(plain.count, padded.count)
        var f = 0
        while f < plain.count - 1 {
            XCTAssertEqual(plain[f], padded[f])
            f += 1
        }
    }
}
