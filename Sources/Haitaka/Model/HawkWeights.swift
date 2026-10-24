import Foundation

/// Default model shape (173M parameters with the attention blocks), shared by training and pretraining.
/// Four times the width the memory budgets are calibrated at (`defaultModelDim`).
public enum HawkShape {
    public static let numLayers = 8
    public static let modelDim = 1152
    /// Width the memory budgets are calibrated at (batch sizes and the MLX buffer cache scale down from it).
    public static let defaultModelDim = 288
    public static let rnnDim = 1536
    public static let mlpDim = 3456
    /// Number of diagonal blocks in the RG-LRU gates.
    public static let heads = 8
    public static let convWidth = 4
    /// Future frames per layer. Total delay is numLayers × convRight frames of 40 ms.
    public static let convRight = 2
    /// Griffin-style local attention blocks, each run after the recurrent layer with this index.
    public static let attentionAfter = [3, 7]
    /// Query heads of the local attention; keys and values share one head (multi-query).
    public static let attentionHeads = 8
    /// Frames each query may look back over (2.56 s at 40 ms), the current frame included. Strictly causal.
    public static let attentionWindow = 64
}

/// Weights of the Hawk acoustic model (the recurrent-only variant of Griffin, De et al. 2024).
/// Each layer is RMSNorm → recurrent block → residual, then RMSNorm → gated MLP → residual.
/// The recurrent block multiplies a GeLU branch with a linear → causal conv → RG-LRU branch, then projects back.
/// All matrices are row-major [input, output] (applied as x · W).
public struct HawkWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let modelDim: Int
    /// Width of the recurrent block.
    public let rnnDim: Int
    /// Hidden width of the MLP.
    public let mlpDim: Int
    /// Number of diagonal blocks in the RG-LRU gate projections.
    public let heads: Int
    public let convWidth: Int
    /// Future frames seen by the convolution (lookahead). Adds this much latency per layer.
    public let convRight: Int
    public let outputDim: Int
    /// Query heads of the local attention blocks (0 when there are none).
    public let attentionHeads: Int
    /// Causal window of the local attention blocks in frames.
    public let attentionWindow: Int

    public let wIn: [Float]          // [inputDim, modelDim]
    /// Input bias. Keeps the residual stream nonzero on all-zero (silent) feature frames; otherwise each
    /// RMSNorm backward pass amplifies gradients by up to 1/√eps and deep stacks overflow.
    public let bIn: [Float]          // [modelDim]
    public let layers: [HawkLayerWeights]
    /// Local attention blocks, in order of `after`.
    public let attention: [HawkAttentionLayerWeights]
    public let finalNorm: [Float]    // [modelDim]
    public let wOut: [Float]         // [modelDim, outputDim]
    public let bOut: [Float]         // [outputDim]
    public let vocabularyCharacters: String?

    /// RG-LRU decay exponent c: a_t = σ(Λ)^(c·r_t).
    public static let decayExponent: Float = 8.0
    /// Epsilon inside RMSNorm, shared by the MLX trainer and the CPU network.
    public static let rmsNormEpsilon: Float = 1e-5
    /// Initial time constants (frames) spread log-uniformly across channels at r = 0.5.
    public static let initialTimeConstants: ClosedRange<Float> = 1.0...100.0

    public init(
        inputDim: Int,
        modelDim: Int,
        rnnDim: Int,
        mlpDim: Int,
        heads: Int,
        convWidth: Int,
        convRight: Int,
        outputDim: Int,
        wIn: [Float],
        bIn: [Float],
        layers: [HawkLayerWeights],
        finalNorm: [Float],
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil,
        attention: [HawkAttentionLayerWeights] = [],
        attentionHeads: Int = 0,
        attentionWindow: Int = 0
    ) {
        self.inputDim = inputDim
        self.attention = attention
        self.attentionHeads = attentionHeads
        self.attentionWindow = attentionWindow
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        self.outputDim = outputDim
        self.wIn = wIn
        self.bIn = bIn
        self.layers = layers
        self.finalNorm = finalNorm
        self.wOut = wOut
        self.bOut = bOut
        self.vocabularyCharacters = vocabularyCharacters
    }

    public var numLayers: Int {
        return layers.count
    }

    /// Weights saved before the attention blocks existed decode with none.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputDim = try c.decode(Int.self, forKey: .inputDim)
        modelDim = try c.decode(Int.self, forKey: .modelDim)
        rnnDim = try c.decode(Int.self, forKey: .rnnDim)
        mlpDim = try c.decode(Int.self, forKey: .mlpDim)
        heads = try c.decode(Int.self, forKey: .heads)
        convWidth = try c.decode(Int.self, forKey: .convWidth)
        convRight = try c.decode(Int.self, forKey: .convRight)
        outputDim = try c.decode(Int.self, forKey: .outputDim)
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 0
        attentionWindow = try c.decodeIfPresent(Int.self, forKey: .attentionWindow) ?? 0
        wIn = try c.decode([Float].self, forKey: .wIn)
        bIn = try c.decode([Float].self, forKey: .bIn)
        layers = try c.decode([HawkLayerWeights].self, forKey: .layers)
        attention = try c.decodeIfPresent([HawkAttentionLayerWeights].self, forKey: .attention) ?? []
        finalNorm = try c.decode([Float].self, forKey: .finalNorm)
        wOut = try c.decode([Float].self, forKey: .wOut)
        bOut = try c.decode([Float].self, forKey: .bOut)
        vocabularyCharacters = try c.decodeIfPresent(String.self, forKey: .vocabularyCharacters)
    }

    /// Attention head width (modelDim / attentionHeads); 0 without attention.
    public var attentionHeadDim: Int {
        if attentionHeads <= 0 {
            return 0
        }
        return modelDim / attentionHeads
    }

    /// Convolution taps: `convWidth - 1` past, the current frame, and `convRight` future.
    public var convTaps: Int {
        return convWidth + convRight
    }

    /// Input-to-output latency in frames.
    public var lookaheadFrames: Int {
        return convRight * numLayers
    }

    public var headDim: Int {
        return rnnDim / heads
    }

    public var parameterCount: Int {
        var n = wIn.count + bIn.count + finalNorm.count + wOut.count + bOut.count
        for layer in layers {
            n += layer.parameterCount
        }
        for block in attention {
            n += block.parameterCount
        }
        return n
    }

    /// Embedded vocabulary, or nil if none was saved.
    public var vocabulary: TextVocabulary? {
        guard let chars = vocabularyCharacters, chars.isEmpty != true else {
            return nil
        }
        return TextVocabulary(serializedCharacters: chars)
    }

    /// Initial Λ such that σ(Λ)^(c/2) = exp(-1/τ) for τ spread over `initialTimeConstants`.
    public static func initialLambda(rnnDim: Int) -> [Float] {
        let lo = logf(initialTimeConstants.lowerBound)
        let hi = logf(initialTimeConstants.upperBound)
        var values = [Float](repeating: 0.0, count: rnnDim)
        var n = 0
        while n < rnnDim {
            var frac: Float = 0.0
            if 1 < rnnDim {
                frac = Float(n) / Float(rnnDim - 1)
            }
            let tau = expf(lo + (hi - lo) * frac)
            let a = expf(-1.0 / (tau * decayExponent * 0.5))
            values[n] = logf(a / (1.0 - a))
            n += 1
        }
        return values
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> HawkWeights {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(HawkWeights.self, from: data)
    }
}

/// Weights of one Hawk layer.
public struct HawkLayerWeights: Sendable, Codable, Equatable {
    public let norm1: [Float]       // [modelDim]
    public let wGate: [Float]       // [modelDim, rnnDim] GeLU branch
    public let wX: [Float]          // [modelDim, rnnDim] recurrent branch
    public let convW: [Float]       // [convWidth + convRight, rnnDim]; row convWidth - 1 is the current frame, later rows are future
    public let convB: [Float]       // [rnnDim]
    public let wA: [Float]          // [heads, headDim, headDim] recurrence gate (block-diagonal)
    public let bA: [Float]          // [rnnDim]
    public let wI: [Float]          // [heads, headDim, headDim] input gate (block-diagonal)
    public let bI: [Float]          // [rnnDim]
    public let lambda: [Float]      // [rnnDim] logit Λ of the decay σ(Λ)
    public let wO: [Float]          // [rnnDim, modelDim]
    public let norm2: [Float]       // [modelDim]
    public let wMLPGate: [Float]    // [modelDim, mlpDim]
    public let wMLPUp: [Float]      // [modelDim, mlpDim]
    public let wMLPDown: [Float]    // [mlpDim, modelDim]

    public init(
        norm1: [Float], wGate: [Float], wX: [Float], convW: [Float], convB: [Float],
        wA: [Float], bA: [Float], wI: [Float], bI: [Float], lambda: [Float], wO: [Float],
        norm2: [Float], wMLPGate: [Float], wMLPUp: [Float], wMLPDown: [Float]
    ) {
        self.norm1 = norm1
        self.wGate = wGate
        self.wX = wX
        self.convW = convW
        self.convB = convB
        self.wA = wA
        self.bA = bA
        self.wI = wI
        self.bI = bI
        self.lambda = lambda
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wGate.count + wX.count + convW.count + convB.count + wA.count + bA.count
            + wI.count + bI.count + lambda.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }
}

/// Weights of one Griffin-style local attention block (RMSNorm → multi-query attention over the last
/// `attentionWindow` frames with rotary positions → residual, then RMSNorm → gated MLP → residual).
public struct HawkAttentionLayerWeights: Sendable, Codable, Equatable {
    /// Index of the recurrent layer this block follows.
    public let after: Int
    public let norm1: [Float]       // [modelDim]
    public let wQ: [Float]          // [modelDim, heads × headDim]
    public let wK: [Float]          // [modelDim, headDim] shared key head
    public let wV: [Float]          // [modelDim, headDim] shared value head
    public let wO: [Float]          // [heads × headDim, modelDim]
    public let norm2: [Float]       // [modelDim]
    public let wMLPGate: [Float]    // [modelDim, mlpDim]
    public let wMLPUp: [Float]      // [modelDim, mlpDim]
    public let wMLPDown: [Float]    // [mlpDim, modelDim]

    public init(after: Int, norm1: [Float], wQ: [Float], wK: [Float], wV: [Float], wO: [Float],
                norm2: [Float], wMLPGate: [Float], wMLPUp: [Float], wMLPDown: [Float]) {
        self.after = after
        self.norm1 = norm1
        self.wQ = wQ
        self.wK = wK
        self.wV = wV
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wQ.count + wK.count + wV.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }

    /// Rotary position angle for pair `i` of `headDim / 2` at frame `position` (base 10000).
    public static func rotaryAngle(position: Int, pair: Int, headDim: Int) -> Float {
        let exponent = -2.0 * Float(pair) / Float(headDim)
        return Float(position) * powf(10000.0, exponent)
    }
}
