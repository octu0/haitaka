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
    /// Reads kana through the encoder and attends to it (off when the decoder listens to the audio only).
    public var kanaInput = true
    /// Width of the acoustic model output the decoder also attends to; 0 when it does not listen.
    public var audioDim = 0

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelDim, rnnDim, mlpDim, heads, convWidth, encoderLayers, encoderAttentionAfter, attentionWindow, decoderLayers
        case kanaInput, audioDim
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelDim = try c.decode(Int.self, forKey: .modelDim)
        rnnDim = try c.decode(Int.self, forKey: .rnnDim)
        mlpDim = try c.decode(Int.self, forKey: .mlpDim)
        heads = try c.decode(Int.self, forKey: .heads)
        convWidth = try c.decode(Int.self, forKey: .convWidth)
        encoderLayers = try c.decode(Int.self, forKey: .encoderLayers)
        encoderAttentionAfter = try c.decode([Int].self, forKey: .encoderAttentionAfter)
        attentionWindow = try c.decode(Int.self, forKey: .attentionWindow)
        decoderLayers = try c.decode(Int.self, forKey: .decoderLayers)
        kanaInput = try c.decodeIfPresent(Bool.self, forKey: .kanaInput) ?? true
        audioDim = try c.decodeIfPresent(Int.self, forKey: .audioDim) ?? 0
    }
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

/// Repetition of a span at the end of a sequence, used to stop decoding loops that the unit's kana does not have.
public enum KanaTextRepetition {
    static let maxPeriod = 32

    /// The most consecutive copies of any span (period ≤ `maxPeriod`) that end `ids[..<end]`.
    public static func tail(_ ids: [Int], end: Int) -> (period: Int, count: Int) {
        var best = (period: 1, count: 1)
        var p = 1
        while p <= min(maxPeriod, end / 2) {
            var c = 1
            while (c + 1) * p <= end {
                var same = true
                var i = 0
                while i < p {
                    if ids[end - p + i] != ids[end - (c + 1) * p + i] {
                        same = false
                        break
                    }
                    i += 1
                }
                if same != true {
                    break
                }
                c += 1
            }
            if best.count < c {
                best = (period: p, count: c)
            }
            p += 1
        }
        return best
    }

    /// A run of one character (digits of 1000, ーーー) counts as a loop only past this many copies, whatever the kana has.
    static let singleRun = 6

    /// Whether the text's tail repeats more than allowed: more copies than `allowed`, and for a single character
    /// also more than `singleRun`.
    public static func exceeds(_ repeated: (period: Int, count: Int), allowed: Int) -> Bool {
        if repeated.period == 1 {
            return max(allowed, singleRun) < repeated.count
        }
        return allowed < repeated.count
    }

    /// The most consecutive copies of any span anywhere in `ids`.
    public static func longest(_ ids: [Int]) -> Int {
        var best = 1
        var end = 2
        while end <= ids.count {
            best = max(best, tail(ids, end: end).count)
            end += 1
        }
        return best
    }
}

/// Cross-attention from the decoder to the encoder outputs of the current unit (or to the acoustic model's output
/// frames, `memoryDim` wide). Pre-RMSNorm, residual. A zero output projection starts it as the identity.
final class MLXKanaTextCrossAttention: Module, @unchecked Sendable {
    public var norm: MLXArray
    public var wQ: MLXArray
    public var wK: MLXArray
    public var wV: MLXArray
    public var wO: MLXArray
    let heads: Int

    init(dim: Int, memoryDim: Int, heads: Int, residualScale: Float, zeroOutput: Bool = false) {
        self.heads = heads
        func uniform(_ fanIn: Int, _ shape: [Int], scale: Float = 1.0) -> MLXArray {
            let bound = scale * sqrt(3.0 / Float(fanIn))
            return MLXRandom.uniform(low: -bound, high: bound, shape)
        }
        norm = MLXArray.ones([dim])
        wQ = uniform(dim, [dim, dim])
        wK = uniform(memoryDim, [memoryDim, dim])
        wV = uniform(memoryDim, [memoryDim, dim])
        if zeroOutput {
            wO = MLXArray.zeros([dim, dim])
        } else {
            wO = uniform(dim, [dim, dim], scale: residualScale)
        }
        super.init()
    }

    /// memory [B, S, memoryDim] → keys and values [B, heads, S, dim/heads].
    func keysValues(_ memory: MLXArray) -> (MLXArray, MLXArray) {
        let b = memory.dim(0)
        let s = memory.dim(1)
        let hd = wK.shape[1] / heads
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
/// With `audioDim`, each decoder layer also attends to the acoustic model's output frames of the utterance (`listen`);
/// with `kanaInput` off it attends to those frames only.
public final class MLXKanaTextModel: Module, @unchecked Sendable {
    public let shape: KanaTextShape
    public var inputEmbedding: MLXArray
    public var outputEmbedding: MLXArray
    var encoder: [MLXHawkLayer]
    var encoderAttention: [MLXHawkAttentionLayer]
    public var encoderNorm: MLXArray
    var decoder: [MLXHawkLayer]
    var cross: [MLXKanaTextCrossAttention]
    var listen: [MLXKanaTextCrossAttention]
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
        var encoderLayers = 0
        var encoderAttentionAfter: [Int] = []
        if shape.kanaInput {
            encoderLayers = shape.encoderLayers
            encoderAttentionAfter = shape.encoderAttentionAfter
        }
        let encoderBlocks = encoderLayers + encoderAttentionAfter.count
        encoder = (0..<encoderLayers).map { _ in hawk(encoderBlocks) }
        encoderAttention = encoderAttentionAfter.map { after in
            MLXHawkAttentionLayer(after: after, modelDim: d, mlpDim: shape.mlpDim, heads: shape.heads,
                                  window: shape.attentionWindow, residualScale: 1.0 / sqrt(Float(2 * encoderBlocks)))
        }
        encoderNorm = MLXArray.ones([d])
        let decoderBlocks = 2 * shape.decoderLayers
        decoder = (0..<shape.decoderLayers).map { _ in hawk(decoderBlocks) }
        var crossLayers = 0
        if shape.kanaInput {
            crossLayers = shape.decoderLayers
        }
        cross = (0..<crossLayers).map { _ in
            MLXKanaTextCrossAttention(dim: d, memoryDim: d, heads: shape.heads, residualScale: 1.0 / sqrt(Float(2 * decoderBlocks)))
        }
        var listenLayers = 0
        if 0 < shape.audioDim {
            listenLayers = shape.decoderLayers
        }
        listen = (0..<listenLayers).map { _ in
            MLXKanaTextCrossAttention(dim: d, memoryDim: shape.audioDim, heads: shape.heads,
                                      residualScale: 1.0 / sqrt(Float(2 * decoderBlocks)), zeroOutput: true)
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

    /// Teacher-forced decoding: previous [B, T] → logits [B, T, V]. `mask` [B, 1, T, S] lets each step see its unit
    /// of `memory`; `audioMask` [B, 1, T or 1, F] does the same for the acoustic frames `audio` [B, F, audioDim].
    public func decode(_ previous: MLXArray, memory: MLXArray?, mask: MLXArray?, audio: MLXArray? = nil,
                       audioMask: MLXArray? = nil) -> MLXArray {
        var h = outputEmbedding[previous]
        var i = 0
        while i < decoder.count {
            h = decoder[i](h, frameMask: nil)
            if let memory, let mask, i < cross.count {
                let (k, v) = cross[i].keysValues(memory)
                h = cross[i](h, keys: k, values: v, mask: mask)
            }
            if let audio, let audioMask, i < listen.count {
                let (k, v) = listen[i].keysValues(audio)
                h = listen[i](h, keys: k, values: v, mask: audioMask)
            }
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

    /// One decoder step: token [B] → logits [B, V]. `mask` [B, 1, 1, S] selects each row's current unit; `audioMask`
    /// [B, 1, 1, F] its acoustic frames.
    func step(_ token: MLXArray, states: inout [HawkStepState], keysValues: [(MLXArray, MLXArray)], mask: MLXArray?,
              audioKeysValues: [(MLXArray, MLXArray)] = [], audioMask: MLXArray? = nil) -> MLXArray {
        var h = outputEmbedding[token]
        var i = 0
        while i < decoder.count {
            h = decoder[i].step(h, state: &states[i])
            if let mask, i < keysValues.count {
                h = cross[i](h.expandedDimensions(axis: 1), keys: keysValues[i].0, values: keysValues[i].1, mask: mask).squeezed(axis: 1)
            }
            if let audioMask, i < audioKeysValues.count {
                h = listen[i](h.expandedDimensions(axis: 1), keys: audioKeysValues[i].0, values: audioKeysValues[i].1,
                              mask: audioMask).squeezed(axis: 1)
            }
            i += 1
        }
        return matmul(hawkRMSNorm(h, decoderNorm), wOut) + bOut
    }

    /// Greedy streaming conversion of a batch of streams. `units[b]` lists the [start, end) kana spans of row b's
    /// units in `source` order; returns the text ids of every unit and how many units were cut short. A unit is cut when
    /// its text ends in more copies of a span than its kana has anywhere (at least two): the extra copies are dropped
    /// and the unit ends there. With `audio` [B, F, audioDim], every unit also listens to its row's frames
    /// (`frameCounts` long).
    public func greedy(_ source: MLXArray, units: [[Range<Int>]], audio: MLXArray? = nil,
                       frameCounts: [Int] = []) -> (units: [[[Int]]], cut: Int) {
        let b = source.dim(0)
        let s = source.dim(1)
        let sourceIds = source.asArray(Int32.self).map { Int($0) }
        let allowed = (0..<b).map { r in
            units[r].map { u in max(2, KanaTextRepetition.longest(Array(sourceIds[(r * s + u.lowerBound)..<(r * s + u.upperBound)]))) }
        }
        var cut = 0
        let memory = encode(source)
        let kv = cross.map { $0.keysValues(memory) }
        eval(memory)
        var audioKV: [(MLXArray, MLXArray)] = []
        var audioMask: MLXArray? = nil
        if let audio, 0 < shape.audioDim {
            audioKV = listen.map { $0.keysValues(audio) }
            audioMask = Self.paddingMask(frameCounts, width: audio.dim(1))
            for (k, v) in audioKV {
                eval(k, v)
            }
        }
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
            let logits = step(MLXArray(token), states: &states, keysValues: kv, mask: mask, audioKeysValues: audioKV,
                              audioMask: audioMask)
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
                        let u = unitIndex[r]
                        outputs[r][u].append(Int(next[r]))
                        let repeated = KanaTextRepetition.tail(outputs[r][u], end: outputs[r][u].count)
                        if KanaTextRepetition.exceeds(repeated, allowed: allowed[r][u]) {
                            outputs[r][u].removeLast((repeated.count - 1) * repeated.period)
                            cut += 1
                            unitIndex[r] += 1
                            produced[r] = 0
                            token[r] = Int32(KanaTextVocabulary.eos)
                        } else {
                            produced[r] += 1
                            token[r] = next[r]
                        }
                    }
                }
                r += 1
            }
        }
        return (units: outputs, cut: cut)
    }

    /// Greedy decoding of one utterance per row, the state starting fresh. `source` [B, S] holds the recognized kana
    /// (`sourceCounts` long) when the model reads kana; `audio` [B, F, audioDim] the acoustic frames (`frameCounts`
    /// long) when it listens. Each row stops at `eos`, after `limits[r]` characters, or when its text ends in more
    /// copies of a span than `allowed[r]` allows (`KanaTextRepetition.exceeds`; the extra copies are dropped).
    public func greedyUtterances(source: MLXArray?, sourceCounts: [Int], audio: MLXArray?, frameCounts: [Int],
                                 limits: [Int], allowed: [Int]) -> (texts: [[Int]], cut: Int) {
        let b = limits.count
        var kv: [(MLXArray, MLXArray)] = []
        var mask: MLXArray? = nil
        if let source, shape.kanaInput {
            let memory = encode(source)
            kv = cross.map { $0.keysValues(memory) }
            mask = Self.paddingMask(sourceCounts, width: source.dim(1))
        }
        var audioKV: [(MLXArray, MLXArray)] = []
        var audioMask: MLXArray? = nil
        if let audio, 0 < shape.audioDim {
            audioKV = listen.map { $0.keysValues(audio) }
            audioMask = Self.paddingMask(frameCounts, width: audio.dim(1))
        }
        for (k, v) in kv + audioKV {
            eval(k, v)
        }
        var states = initialStates(b)
        var outputs = [[Int]](repeating: [], count: b)
        var done = limits.map { $0 <= 0 }
        var cut = 0
        var token = [Int32](repeating: Int32(KanaTextVocabulary.bos), count: b)
        while done.contains(false) {
            let logits = step(MLXArray(token), states: &states, keysValues: kv, mask: mask, audioKeysValues: audioKV,
                              audioMask: audioMask)
            let next = argMax(logits, axis: -1).asArray(Int32.self)
            for state in states {
                eval(state.h, state.conv)
            }
            var r = 0
            while r < b {
                if done[r] != true {
                    if next[r] == Int32(KanaTextVocabulary.eos) || limits[r] <= outputs[r].count {
                        done[r] = true
                    } else {
                        outputs[r].append(Int(next[r]))
                        let repeated = KanaTextRepetition.tail(outputs[r], end: outputs[r].count)
                        if KanaTextRepetition.exceeds(repeated, allowed: allowed[r]) {
                            outputs[r].removeLast((repeated.count - 1) * repeated.period)
                            cut += 1
                            done[r] = true
                        }
                    }
                    token[r] = next[r]
                }
                r += 1
            }
        }
        return (texts: outputs, cut: cut)
    }

    /// Additive mask [B, 1, 1, width] that hides positions at or past each row's count.
    public static func paddingMask(_ counts: [Int], width: Int) -> MLXArray {
        var values = [Float](repeating: -1e9, count: counts.count * width)
        var r = 0
        while r < counts.count {
            var p = 0
            while p < min(max(1, counts[r]), width) {
                values[r * width + p] = 0.0
                p += 1
            }
            r += 1
        }
        return MLXArray(values, [counts.count, 1, 1, width])
    }
}
