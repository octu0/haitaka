import XCTest
import MLX
@testable import Haitaka

/// The streaming kana-to-text converter: unit splitting, and step-by-step decoding matching teacher-forced decoding.
final class KanaTextModelTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .gpu)
        MLXRandom.seed(3)
    }

    func testUnitsSplitAtPunctuationOnBothSides() {
        let units = KanaTextUnits.paired(kana: "きょーわ、いいてんきだね。ほんと！", text: "今日は、いい天気だね。ほんと!")
        XCTAssertEqual(units.map { $0.kana }, ["きょーわ、", "いいてんきだね。", "ほんと！"])
        XCTAssertEqual(units.map { $0.text }, ["今日は、", "いい天気だね。", "ほんと!"])
    }

    func testDecimalPointAndMarkRunsDoNotSplit() {
        XCTAssertEqual(KanaTextUnits.splitText("2.3キロ走った!?すごい"), ["2.3キロ走った!?", "すごい"])
        XCTAssertEqual(KanaTextUnits.splitKana("にーてんさんきろはしった！すごい"), ["にーてんさんきろはしった！", "すごい"])
    }

    /// Lines whose two sides split differently stay whole.
    func testMismatchedUnitsKeepTheLineWhole() {
        let units = KanaTextUnits.paired(kana: "あ、い", text: "あい")
        XCTAssertEqual(units.count, 1)
        XCTAssertEqual(units[0].text, "あい")
    }

    func testStepDecodingMatchesTeacherForcedDecoding() {
        var shape = KanaTextShape()
        shape.modelDim = 32
        shape.rnnDim = 48
        shape.mlpDim = 64
        shape.heads = 4
        shape.encoderLayers = 3
        shape.encoderAttentionAfter = [1]
        shape.attentionWindow = 5
        shape.decoderLayers = 2
        let model = MLXKanaTextModel(shape: shape, inputVocabulary: 11, outputVocabulary: 13)
        let source = MLXArray([4, 5, 6, 7, 8, 9, 10, 4, 5] as [Int32], [1, 9])
        let sourceUnit = [0, 0, 0, 1, 1, 1, 1, 2, 2]
        let previous: [Int32] = [1, 5, 6, 2, 7, 8, 9, 2, 10, 4]
        let targetUnit = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        var maskValues = [Float](repeating: -1e9, count: previous.count * sourceUnit.count)
        var t = 0
        while t < previous.count {
            var s = 0
            while s < sourceUnit.count {
                if sourceUnit[s] == targetUnit[t] {
                    maskValues[t * sourceUnit.count + s] = 0.0
                }
                s += 1
            }
            t += 1
        }
        let memory = model.encode(source)
        let full = model.decode(MLXArray(previous, [1, previous.count]), memory: memory,
                                mask: MLXArray(maskValues, [1, 1, previous.count, sourceUnit.count]))
        let kv = model.cross.map { $0.keysValues(memory) }
        var states = model.initialStates(1)
        var worst: Float = 0.0
        t = 0
        while t < previous.count {
            let row = MLXArray(Array(maskValues[(t * sourceUnit.count)..<((t + 1) * sourceUnit.count)]), [1, 1, 1, sourceUnit.count])
            let logits = model.step(MLXArray([previous[t]]), states: &states, keysValues: kv, mask: row)
            worst = max(worst, abs(logits - full[0..., t, 0...]).max().item(Float.self))
            t += 1
        }
        XCTAssertLessThan(worst, 1e-4)
    }
}
