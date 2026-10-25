import Foundation
import MLX
import MLXNN

/// Character vocabularies of the kana-to-text model. Index 0 pads, 1 starts and 2 ends a sequence, 3 stands for
/// characters missing from the vocabulary.
public struct KanaTextVocabulary: Codable, Sendable {
    public static let pad = 0
    public static let bos = 1
    public static let eos = 2
    public static let unknown = 3
    static let specialCount = 4

    public let input: [String]
    public let output: [String]
    private var inputIndex: [String: Int] = [:]
    private var outputIndex: [String: Int] = [:]

    enum CodingKeys: String, CodingKey {
        case input
        case output
    }

    /// Output characters seen fewer than `minOutputCount` times map to `unknown`.
    public init(pairs: [(kana: String, text: String)], minOutputCount: Int = 1) {
        var kanaCounts: [String: Int] = [:]
        var textCounts: [String: Int] = [:]
        for p in pairs {
            for c in p.kana {
                kanaCounts[String(c), default: 0] += 1
            }
            for c in p.text {
                textCounts[String(c), default: 0] += 1
            }
        }
        func ordered(_ counts: [String: Int], _ minCount: Int) -> [String] {
            let kept = counts.filter { minCount <= $0.value }
            return [String](repeating: "", count: Self.specialCount) + kept.keys.sorted { a, b in
                if kept[a]! != kept[b]! {
                    return kept[b]! < kept[a]!
                }
                return a < b
            }
        }
        self.init(input: ordered(kanaCounts, 1), output: ordered(textCounts, minOutputCount))
    }

    public init(input: [String], output: [String]) {
        self.input = input
        self.output = output
        buildIndex()
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        input = try c.decode([String].self, forKey: .input)
        output = try c.decode([String].self, forKey: .output)
        buildIndex()
    }

    private mutating func buildIndex() {
        var i = Self.specialCount
        while i < input.count {
            inputIndex[input[i]] = i
            i += 1
        }
        i = Self.specialCount
        while i < output.count {
            outputIndex[output[i]] = i
            i += 1
        }
    }

    public func encodeInput(_ kana: String) -> [Int] {
        return kana.map { inputIndex[String($0)] ?? Self.unknown }
    }

    public func encodeOutput(_ text: String) -> [Int] {
        return text.map { outputIndex[String($0)] ?? Self.unknown }
    }

    /// Stops at the end token; unknown characters are dropped.
    public func decodeOutput(_ ids: [Int]) -> String {
        var s = ""
        for id in ids {
            if id == Self.eos {
                break
            }
            if Self.specialCount <= id && id < output.count {
                s += output[id]
            }
        }
        return s
    }
}

public struct KanaTextShape: Codable, Sendable {
    public var modelDim = 384
    public var rnnDim = 384
    public var mlpDim = 1152
    public var heads = 6
    public var convWidth = 4
    public var encoderLayers = 6
    public var encoderAttentionAfter = [2, 5]
    public var attentionWindow = 64
    public var decoderLayers = 6

    public init() {}
}

/// Splits a line into units that end at punctuation, kana and text alike. Returns nil when the two sides do not
/// split into the same number of units (the caller keeps the line whole).
public enum KanaTextUnits {
    static let kanaMarks: Set<Character> = ["、", "。", "？", "！"]
    static let textMarks: Set<Character> = ["、", "，", "。", "．", "？", "！", "?", "!", ",", "."]

    public static func splitKana(_ kana: String) -> [String] {
        var units: [String] = []
        var current = ""
        for c in kana {
            current.append(c)
            if kanaMarks.contains(c) {
                units.append(current)
                current = ""
            }
        }
        if current.isEmpty != true {
            units.append(current)
        }
        return units
    }

    /// A run of marks ends one unit; ASCII , and . count only when no letter or digit follows (2.3, Mr.Smith).
    public static func splitText(_ text: String) -> [String] {
        let chars = Array(text)
        var units: [String] = []
        var current = ""
        var i = 0
        while i < chars.count {
            let c = chars[i]
            current.append(c)
            var isMark = textMarks.contains(c)
            if c == "," || c == "." {
                if i + 1 < chars.count && (chars[i + 1].isLetter || chars[i + 1].isNumber) {
                    isMark = false
                }
            }
            var nextIsMark = false
            if i + 1 < chars.count {
                nextIsMark = textMarks.contains(chars[i + 1])
            }
            let hasContent = current.contains { textMarks.contains($0) != true }
            if isMark && nextIsMark != true && hasContent {
                units.append(current)
                current = ""
            }
            i += 1
        }
        if current.isEmpty != true {
            if current.contains(where: { textMarks.contains($0) != true }) || units.isEmpty {
                units.append(current)
            } else {
                units[units.count - 1] += current
            }
        }
        return units
    }

    public static func paired(kana: String, text: String) -> [(kana: String, text: String)] {
        let k = splitKana(kana)
        let t = splitText(text)
        if k.count == t.count {
            return Array(zip(k, t)).map { (kana: $0.0, text: $0.1) }
        }
        return [(kana: kana, text: text)]
    }
}

/// Cross-attention from the decoder to the encoder outputs of the current unit. Pre-RMSNorm, residual.
final class MLXKanaTextCrossAttention: Module, @unchecked Sendable {
    public var norm: MLXArray
    public var wQ: MLXArray
    public var wK: MLXArray
    public var wV: MLXArray
    public var wO: MLXArray
    let heads: Int

    init(dim: Int, heads: Int, residualScale: Float) {
        self.heads = heads
        func uniform(_ fanIn: Int, _ shape: [Int], scale: Float = 1.0) -> MLXArray {
            let bound = scale * sqrt(3.0 / Float(fanIn))
            return MLXRandom.uniform(low: -bound, high: bound, shape)
        }
        norm = MLXArray.ones([dim])
        wQ = uniform(dim, [dim, dim])
        wK = uniform(dim, [dim, dim])
        wV = uniform(dim, [dim, dim])
        wO = uniform(dim, [dim, dim], scale: residualScale)
        super.init()
    }

    /// memory [B, S, D] → keys and values [B, heads, S, D/heads].
    func keysValues(_ memory: MLXArray) -> (MLXArray, MLXArray) {
        let b = memory.dim(0)
        let s = memory.dim(1)
        let hd = memory.dim(2) / heads
        let k = matmul(memory, wK).reshaped([b, s, heads, hd]).transposed(0, 2, 1, 3)
        let v = matmul(memory, wV).reshaped([b, s, heads, hd]).transposed(0, 2, 1, 3)
        return (k, v)
    }

    /// x [B, T, D]; mask additive [B, 1, T, S].
    func callAsFunction(_ x: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let d = x.dim(2)
        let hd = d / heads
        let q = matmul(hawkRMSNorm(x, norm), wQ).reshaped([b, t, heads, hd]).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, scale: 1.0 / sqrt(Float(hd)), mask: mask)
        return x + matmul(out.transposed(0, 2, 1, 3).reshaped([b, t, d]), wO)
    }
}

/// Recurrent state of one causal Hawk layer for step-by-step decoding.
public struct HawkStepState {
    var h: MLXArray
    var conv: MLXArray
}

extension MLXHawkLayer {
    /// One frame of `callAsFunction` (causal layers only): x [B, D] → [B, D].
    func step(_ x: MLXArray, state: inout HawkStepState) -> MLXArray {
        let batch = x.dim(0)
        let rnnDim = wX.shape[1]
        let headDim = rnnDim / heads
        let n = hawkRMSNorm(x, norm1)
        let branch = hawkGelu(matmul(n, wGate))
        let window = concatenated([state.conv, matmul(n, wX).expandedDimensions(axis: 1)], axis: 1)
        let u = convB + sum(window * convW, axis: 1)
        let blocks = u.reshaped([batch, heads, headDim]).transposed(1, 0, 2)
        func gate(_ w: MLXArray, _ bias: MLXArray) -> MLXArray {
            return sigmoid(matmul(blocks, w).transposed(1, 0, 2).reshaped([batch, rnnDim]) + bias)
        }
        let r = gate(wA, bA)
        let i = gate(wI, bI)
        let logA = (HawkWeights.decayExponent * r) * logSigmoid(lambda)
        let h = exp(logA) * state.h + sqrt(1.0 - exp(2.0 * logA) + 1e-6) * (i * u)
        state.h = h
        state.conv = window[0..., 1..., 0...]
        let mixed = x + matmul(branch * h, wO)
        let m = hawkRMSNorm(mixed, norm2)
        return mixed + matmul(hawkGelu(matmul(m, wMLPGate)) * matmul(m, wMLPUp), wMLPDown)
    }
}

/// Streaming kana-to-text converter built from Hawk layers. The encoder reads pronunciation kana (the acoustic model's
/// targets) left to right; the decoder writes text one character per step and attends only to the encoder outputs of
/// the current unit (kana up to a punctuation mark). Both carry their recurrent state across units, so context reaches
/// back over the whole stream while memory stays constant. Each unit's text ends with `eos`, which also starts the next.
public final class MLXKanaTextModel: Module, @unchecked Sendable {
    public let shape: KanaTextShape
    public var inputEmbedding: MLXArray
    public var outputEmbedding: MLXArray
    var encoder: [MLXHawkLayer]
    var encoderAttention: [MLXHawkAttentionLayer]
    public var encoderNorm: MLXArray
    var decoder: [MLXHawkLayer]
    var cross: [MLXKanaTextCrossAttention]
    public var decoderNorm: MLXArray
    public var wOut: MLXArray
    public var bOut: MLXArray

    public init(shape: KanaTextShape, inputVocabulary: Int, outputVocabulary: Int) {
        self.shape = shape
        let d = shape.modelDim
        inputEmbedding = MLXRandom.normal([inputVocabulary, d]) * 0.5
        outputEmbedding = MLXRandom.normal([outputVocabulary, d]) * 0.5
        func hawk(_ layers: Int) -> MLXHawkLayer {
            return MLXHawkLayer(modelDim: d, rnnDim: shape.rnnDim, mlpDim: shape.mlpDim, heads: shape.heads,
                                convWidth: shape.convWidth, convRight: 0, numLayers: layers)
        }
        let encoderBlocks = shape.encoderLayers + shape.encoderAttentionAfter.count
        encoder = (0..<shape.encoderLayers).map { _ in hawk(encoderBlocks) }
        encoderAttention = shape.encoderAttentionAfter.map { after in
            MLXHawkAttentionLayer(after: after, modelDim: d, mlpDim: shape.mlpDim, heads: shape.heads,
                                  window: shape.attentionWindow, residualScale: 1.0 / sqrt(Float(2 * encoderBlocks)))
        }
        encoderNorm = MLXArray.ones([d])
        let decoderBlocks = 2 * shape.decoderLayers
        decoder = (0..<shape.decoderLayers).map { _ in hawk(decoderBlocks) }
        cross = (0..<shape.decoderLayers).map { _ in
            MLXKanaTextCrossAttention(dim: d, heads: shape.heads, residualScale: 1.0 / sqrt(Float(2 * decoderBlocks)))
        }
        decoderNorm = MLXArray.ones([d])
        let bound = sqrt(3.0 / Float(d))
        wOut = MLXRandom.uniform(low: -bound, high: bound, [d, outputVocabulary])
        bOut = MLXArray.zeros([outputVocabulary])
        super.init()
    }

    /// source [B, S] kana ids → memory [B, S, D].
    public func encode(_ source: MLXArray) -> MLXArray {
        var h = inputEmbedding[source]
        var i = 0
        while i < encoder.count {
            h = encoder[i](h, frameMask: nil)
            for a in encoderAttention where a.after == i {
                h = a(h)
            }
            i += 1
        }
        return hawkRMSNorm(h, encoderNorm)
    }

    /// Teacher-forced decoding: previous [B, T] → logits [B, T, V]. `mask` [B, 1, T, S] lets each step see its unit.
    public func decode(_ previous: MLXArray, memory: MLXArray, mask: MLXArray) -> MLXArray {
        var h = outputEmbedding[previous]
        var i = 0
        while i < decoder.count {
            h = decoder[i](h, frameMask: nil)
            let (k, v) = cross[i].keysValues(memory)
            h = cross[i](h, keys: k, values: v, mask: mask)
            i += 1
        }
        return matmul(hawkRMSNorm(h, decoderNorm), wOut) + bOut
    }

    /// Label-smoothed cross entropy averaged over non-padding target positions.
    public static func loss(logits: MLXArray, targets: MLXArray, smoothing: Float) -> MLXArray {
        let logp = logSoftmax(logits, axis: -1)
        let nll = -takeAlong(logp, targets.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let uniform = -mean(logp, axis: -1)
        let valid = (targets .!= MLXArray(Int32(KanaTextVocabulary.pad))).asType(.float32)
        let perToken = (1.0 - smoothing) * nll + smoothing * uniform
        return sum(perToken * valid) / maximum(sum(valid), MLXArray(Float(1.0)))
    }

    /// Decoder states for `b` streams before their first step.
    func initialStates(_ b: Int) -> [HawkStepState] {
        return decoder.map { _ in
            HawkStepState(h: MLXArray.zeros([b, shape.rnnDim]), conv: MLXArray.zeros([b, shape.convWidth - 1, shape.rnnDim]))
        }
    }

    /// One decoder step: token [B] → logits [B, V]. `mask` [B, 1, 1, S] selects each row's current unit.
    func step(_ token: MLXArray, states: inout [HawkStepState], keysValues: [(MLXArray, MLXArray)], mask: MLXArray) -> MLXArray {
        var h = outputEmbedding[token]
        var i = 0
        while i < decoder.count {
            h = decoder[i].step(h, state: &states[i])
            h = cross[i](h.expandedDimensions(axis: 1), keys: keysValues[i].0, values: keysValues[i].1, mask: mask).squeezed(axis: 1)
            i += 1
        }
        return matmul(hawkRMSNorm(h, decoderNorm), wOut) + bOut
    }

    /// Greedy streaming conversion of a batch of streams. `units[b]` lists the [start, end) kana spans of row b's
    /// units in `source` order; returns the text ids of every unit.
    public func greedy(_ source: MLXArray, units: [[Range<Int>]]) -> [[[Int]]] {
        let b = source.dim(0)
        let s = source.dim(1)
        let memory = encode(source)
        let kv = cross.map { $0.keysValues(memory) }
        eval(memory)
        var states = initialStates(b)
        var unitIndex = [Int](repeating: 0, count: b)
        var produced = [Int](repeating: 0, count: b)
        var outputs = units.map { u in [[Int]](repeating: [], count: u.count) }
        var token = [Int32](repeating: Int32(KanaTextVocabulary.bos), count: b)
        func active(_ r: Int) -> Bool {
            return unitIndex[r] < units[r].count
        }
        while (0..<b).contains(where: active) {
            var maskValues = [Float](repeating: -1e9, count: b * s)
            var r = 0
            while r < b {
                if active(r) {
                    for p in units[r][unitIndex[r]] {
                        maskValues[r * s + p] = 0.0
                    }
                } else {
                    maskValues[r * s] = 0.0
                }
                r += 1
            }
            let mask = MLXArray(maskValues, [b, 1, 1, s])
            let logits = step(MLXArray(token), states: &states, keysValues: kv, mask: mask)
            let next = argMax(logits, axis: -1).asArray(Int32.self)
            for state in states {
                eval(state.h, state.conv)
            }
            r = 0
            while r < b {
                if active(r) {
                    let limit = 2 * units[r][unitIndex[r]].count + 8
                    if next[r] == Int32(KanaTextVocabulary.eos) || limit <= produced[r] {
                        unitIndex[r] += 1
                        produced[r] = 0
                        token[r] = Int32(KanaTextVocabulary.eos)
                    } else {
                        outputs[r][unitIndex[r]].append(Int(next[r]))
                        produced[r] += 1
                        token[r] = next[r]
                    }
                }
                r += 1
            }
        }
        return outputs
    }
}
