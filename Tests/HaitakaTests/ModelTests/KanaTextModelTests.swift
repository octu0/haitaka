import XCTest
import MLX
import MLXNN
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

    func testRepetitionCountsCopiesOfTheTailSpan() {
        let t = KanaTextRepetition.tail([9, 1, 2, 3, 1, 2, 3, 1, 2, 3], end: 10)
        XCTAssertEqual(t.period, 3)
        XCTAssertEqual(t.count, 3)
        XCTAssertEqual(KanaTextRepetition.tail([1, 2, 3, 4], end: 4).count, 1)
        XCTAssertEqual(KanaTextRepetition.longest([5, 6, 5, 6, 7, 7, 7, 7, 8]), 4)
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

    private func smallShape() -> KanaTextShape {
        var shape = KanaTextShape()
        shape.modelDim = 32
        shape.rnnDim = 48
        shape.mlpDim = 64
        shape.heads = 4
        shape.encoderLayers = 2
        shape.encoderAttentionAfter = [0]
        shape.attentionWindow = 5
        shape.decoderLayers = 2
        return shape
    }

    /// A shape saved before the decoder could listen reads back as kana only.
    func testShapeWithoutListenFieldsDecodesAsKanaOnly() throws {
        let json = #"{"modelDim":384,"rnnDim":384,"mlpDim":1152,"heads":6,"convWidth":4,"encoderLayers":6,"encoderAttentionAfter":[2,5],"attentionWindow":64,"decoderLayers":6}"#
        let shape = try JSONDecoder().decode(KanaTextShape.self, from: Data(json.utf8))
        XCTAssertTrue(shape.kanaInput)
        XCTAssertEqual(shape.audioDim, 0)
    }

    /// The frame attention starts as the identity, so a kana model's weights give the same output whatever it hears.
    func testZeroInitializedListenKeepsTheKanaModelOutput() {
        let kanaOnly = MLXKanaTextModel(shape: smallShape(), inputVocabulary: 11, outputVocabulary: 13)
        var listening = smallShape()
        listening.audioDim = 7
        let both = MLXKanaTextModel(shape: listening, inputVocabulary: 11, outputVocabulary: 13)
        let own = Dictionary(uniqueKeysWithValues: both.parameters().flattened())
        let shared = Dictionary(uniqueKeysWithValues: kanaOnly.parameters().flattened()).filter { own[$0.key]?.shape == $0.value.shape }
        both.update(parameters: ModuleParameters.unflattened(shared))
        let source = MLXArray([4, 5, 6, 7] as [Int32], [1, 4])
        let previous = MLXArray([1, 5, 6, 8] as [Int32], [1, 4])
        let mask = MLXKanaTextModel.paddingMask([4], width: 4)
        let audio = MLXRandom.normal([1, 6, 7])
        let expected = kanaOnly.decode(previous, memory: kanaOnly.encode(source), mask: mask)
        let heard = both.decode(previous, memory: both.encode(source), mask: mask, audio: audio,
                                audioMask: MLXKanaTextModel.paddingMask([5], width: 6))
        XCTAssertLessThan(abs(heard - expected).max().item(Float.self), 1e-5)
    }

    /// Listening only: stepping one character at a time matches teacher forcing, and padded frames are ignored.
    func testListenStepDecodingMatchesTeacherForcedDecoding() {
        var shape = smallShape()
        shape.kanaInput = false
        shape.audioDim = 7
        let model = MLXKanaTextModel(shape: shape, inputVocabulary: 11, outputVocabulary: 13)
        for l in model.listen {
            l.wO = MLXRandom.normal(l.wO.shape) * 0.1
        }
        XCTAssertTrue(model.cross.isEmpty)
        let previous: [Int32] = [1, 5, 6, 7, 9]
        let frames = MLXRandom.normal([1, 6, 7])
        let padded = concatenated([frames, MLXRandom.normal([1, 3, 7]) * 100.0], axis: 1)
        let full = model.decode(MLXArray(previous, [1, previous.count]), memory: nil, mask: nil, audio: frames,
                                audioMask: MLXKanaTextModel.paddingMask([6], width: 6))
        let audioMask = MLXKanaTextModel.paddingMask([6], width: 9)
        let kv = model.listen.map { $0.keysValues(padded) }
        var states = model.initialStates(1)
        var worst: Float = 0.0
        var t = 0
        while t < previous.count {
            let logits = model.step(MLXArray([previous[t]]), states: &states, keysValues: [], mask: nil, audioKeysValues: kv,
                                    audioMask: audioMask)
            worst = max(worst, abs(logits - full[0..., t, 0...]).max().item(Float.self))
            t += 1
        }
        XCTAssertLessThan(worst, 1e-4)
    }
}
