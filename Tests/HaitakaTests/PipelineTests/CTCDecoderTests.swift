import Foundation
import XCTest
@testable import Haitaka

final class CTCDecoderTests: XCTestCase {
    /// A token held over several frames collapses to a single output.
    func testBeamSearchCollapsesRepeats() {
        let vocab = TextVocabulary(corpus: ["あいうえお", "かきくけこ"])
        let decoder = CTCDecoder(vocabulary: vocab, beamWidth: 8)

        let tCount = 10
        let vCount = vocab.size

        // Token 4 (あ) dominates frames 2-4.
        var logProbs = [[Float]](repeating: [Float](repeating: -10.0, count: vCount), count: tCount)
        for t in 0..<tCount {
            logProbs[t][0] = 0.0 // Blank
        }
        logProbs[2][4] = 5.0
        logProbs[3][4] = 5.0
        logProbs[4][4] = 5.0

        // Log-softmax per frame.
        for t in 0..<tCount {
            var maxL: Float = -.infinity
            for v in 0..<vCount {
                if maxL < logProbs[t][v] { maxL = logProbs[t][v] }
            }
            var sumExp: Float = 0.0
            for v in 0..<vCount {
                sumExp += exp(logProbs[t][v] - maxL)
            }
            let lse = maxL + log(sumExp)
            for v in 0..<vCount {
                logProbs[t][v] -= lse
            }
        }

        let beamRes = decoder.decode(logProbs: logProbs)
        XCTAssertEqual(beamRes.tokens, [4], "Beam search CTC should decode identical token")
    }

    /// Feeding frames one at a time matches batch decoding.
    func testCTCStreamingMatchesBatch() {
        let vocab = TextVocabulary(characters: ["あ", "い", "う", "え", "お"])
        let decoder = CTCDecoder(vocabulary: vocab, beamWidth: 8)

        // Arbitrary but deterministic log-probabilities.
        var logProbs: [[Float]] = []
        var t = 0
        while t < 40 {
            var frame = [Float](repeating: -8.0, count: vocab.size)
            let peak = (t / 5) % vocab.size
            frame[peak] = -0.2
            frame[(peak + 1) % vocab.size] = -1.5
            frame[0] = (t % 3 == 0) ? -0.5 : -3.0
            logProbs.append(frame)
            t += 1
        }

        let batch = decoder.decode(logProbs: logProbs)

        let streaming = CTCDecoder(vocabulary: vocab, beamWidth: 8)
        var i = 0
        while i < logProbs.count {
            streaming.push(frame: logProbs[i])
            i += 1
        }
        let incremental = streaming.best

        XCTAssertEqual(streaming.frameCount, logProbs.count)
        XCTAssertEqual(incremental.text, batch.text)
        XCTAssertEqual(incremental.tokens, batch.tokens)
        XCTAssertEqual(incremental.score, batch.score, accuracy: 1e-5)
    }

    /// Partial results grow, and reset returns to the initial state.
    func testCTCStreamingPartialAndReset() {
        let vocab = TextVocabulary(characters: ["あ", "い", "う"])
        let decoder = CTCDecoder(vocabulary: vocab, beamWidth: 4)
        let streaming = decoder

        XCTAssertEqual(streaming.best.text, "")

        // Frames dominated by あ make the partial result non-empty.
        var frame = [Float](repeating: -9.0, count: vocab.size)
        frame[vocab.id(for: "あ")] = -0.05
        streaming.push(frame: frame)
        streaming.push(frame: frame)
        let partial = streaming.best.text
        XCTAssertFalse(partial.isEmpty)

        streaming.reset()
        XCTAssertEqual(streaming.frameCount, 0)
        XCTAssertEqual(streaming.best.text, "")
    }
}
