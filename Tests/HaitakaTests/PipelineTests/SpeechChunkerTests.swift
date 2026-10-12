import XCTest
@testable import Haitaka

final class SpeechChunkerTests: XCTestCase {
    private let sampleRate = 16000

    /// Synthetic voiced signal (fundamental plus harmonics, light noise) lasting `seconds`.
    private func voiced(seconds: Float, seed: inout UInt32) -> [Float] {
        let n = Int(seconds * Float(sampleRate))
        var out = [Float](repeating: 0.0, count: n)
        var i = 0
        while i < n {
            let t = Float(i) / Float(sampleRate)
            var v = 0.5 * sin(2.0 * Float.pi * 140.0 * t)
            v += 0.25 * sin(2.0 * Float.pi * 280.0 * t)
            v += 0.12 * sin(2.0 * Float.pi * 420.0 * t)
            seed = seed &* 1664525 &+ 1013904223
            v += (Float(seed >> 8) / Float(1 << 24) - 0.5) * 0.02
            out[i] = v * 0.3
            i += 1
        }
        return out
    }

    private func silence(seconds: Float) -> [Float] {
        return [Float](repeating: 0.0, count: Int(seconds * Float(sampleRate)))
    }

    func testSplitsAtSilenceAndDropsShortBursts() {
        var seed: UInt32 = 7
        var pcm: [Float] = []
        pcm += silence(seconds: 1.0)
        pcm += voiced(seconds: 2.0, seed: &seed)
        pcm += silence(seconds: 1.5)
        pcm += voiced(seconds: 3.0, seed: &seed)
        pcm += silence(seconds: 1.5)
        pcm += voiced(seconds: 0.03, seed: &seed)  // Shorter than the VAD minimum (50 ms), so dropped
        pcm += silence(seconds: 1.0)

        let spans = SpeechChunker().chunk(pcm: pcm)
        XCTAssertEqual(spans.count, 2)
        if spans.count == 2 {
            // First span covers about 1.0-3.0 s, including 150 ms of padding on each side.
            XCTAssertLessThan(abs(Float(spans[0].start) / 16000.0 - 0.85), 0.3)
            XCTAssertLessThan(abs(Float(spans[0].end) / 16000.0 - 3.15), 0.4)
            XCTAssertLessThan(spans[0].end, spans[1].start)
            XCTAssertLessThan(abs(Float(spans[1].end - spans[1].start) / 16000.0 - 3.3), 0.5)
        }
    }

    func testMergesShortGapsAndCapsLength() {
        var seed: UInt32 = 3
        var pcm: [Float] = []
        pcm += silence(seconds: 0.5)
        // Utterances separated by 0.2 s gaps merge, then split at the 4 s cap.
        var k = 0
        while k < 6 {
            pcm += voiced(seconds: 1.5, seed: &seed)
            pcm += silence(seconds: 0.2)
            k += 1
        }
        pcm += silence(seconds: 1.0)

        let spans = SpeechChunker(maxSegmentSeconds: 4.0).chunk(pcm: pcm)
        XCTAssertLessThan(1, spans.count)
        var total: Float = 0.0
        var i = 0
        while i < spans.count {
            let seconds = Float(spans[i].end - spans[i].start) / 16000.0
            XCTAssertLessThanOrEqual(seconds, 4.0 + 0.01)
            if 0 < i {
                XCTAssertLessThanOrEqual(spans[i - 1].end, spans[i].start)
            }
            total += seconds
            i += 1
        }
        // Nearly all of the 9 s of speech is kept.
        XCTAssertLessThan(8.0, total)
    }

    func testEmptyOrSilentInputGivesNoSpans() {
        XCTAssertEqual(SpeechChunker().chunk(pcm: []).count, 0)
        XCTAssertEqual(SpeechChunker().chunk(pcm: silence(seconds: 3.0)).count, 0)
    }
}
