import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// CTC training for the Hawk acoustic model. RG-LRU is a linear recurrence, so gradients flow through the whole sequence without truncation.
public final class MLXHawkTrainer: @unchecked Sendable {
    public let network: MLXHawkNetwork
    public let optimizer: ArrayLRAdam

    /// Gradient norm of the most recent batch, before clipping.
    public private(set) var lastGradientNorm: Float = 0.0

    /// Compiled training steps keyed by padded sequence length (frames rounded up to a multiple of 32).
    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// MLX buffer cache limit. Each padded sequence length needs its own buffers, so the cache grows without
    /// bound; a limit below one step's working set forces reallocation every step and slows training.
    public static let cacheLimitBytes = 24 << 30

    public init(network: MLXHawkNetwork, learningRate: Float) {
        self.network = network
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        MLX.GPU.set(cacheLimit: Self.cacheLimitBytes)
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// Restores saved weights and resets the Adam moments.
    public func rollback(to weights: HawkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        compiledSteps.removeAll()
        eval(network, optimizer)
    }

    /// Inference logits [B, T, outputDim].
    public func logits(features: MLXArray) -> MLXArray {
        return network(features)
    }

    /// Runs one CTC training step and returns the loss.
    public func trainBatchCTC(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        blankId: Int = 0,
        compiled: Bool = true
    ) -> Float {
        // CTC is undefined for empty targets; skip those samples.
        var validFeatures: [[[Float]]] = []
        var validTargets: [[Int]] = []
        var frameCounts: [Int] = []
        var maxT = 0
        var b = 0
        while b < featuresBatch.count {
            let frames = featuresBatch[b].count
            if 0 < frames && targetsBatch[b].isEmpty != true {
                validFeatures.append(featuresBatch[b])
                validTargets.append(targetsBatch[b])
                frameCounts.append(frames)
                maxT = max(maxT, frames)
            }
            b += 1
        }
        if validFeatures.isEmpty {
            return 0.0
        }
        // Round T up to a multiple of 32 to limit distinct compiled shapes; padded frames are masked by inputLengths.
        maxT = ((maxT + 31) / 32) * 32
        let count = validFeatures.count
        let inDim = network.inputDim
        var flat = [Float](repeating: 0.0, count: count * maxT * inDim)
        b = 0
        while b < count {
            var t = 0
            while t < validFeatures[b].count {
                let offset = (b * maxT + t) * inDim
                let row = validFeatures[b][t]
                var d = 0
                while d < inDim {
                    flat[offset + d] = row[d]
                    d += 1
                }
                t += 1
            }
            b += 1
        }
        let features = MLXArray(flat, [count, maxT, inDim])
        let targets = MLXCTCLoss.ExtendedTargets(targetsBatch: validTargets, frameCounts: frameCounts, blankId: blankId)

        let lossFn = { (model: MLXHawkNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let ext = MLXCTCLoss.ExtendedTargets(
                extTargets: arrays[1],
                skipMask: arrays[2],
                validMask: arrays[3],
                finalIndex1: arrays[4],
                finalIndex2: arrays[5],
                hasSecondFinal: arrays[6],
                inputLengths: arrays[7]
            )
            return [MLXCTCLoss.loss(logits: model(arrays[0], frameCounts: arrays[7]), targets: ext)]
        }
        let lg = valueAndGrad(model: network, lossFn)
        // The learning rate is passed as the last input so the compiled graph reads it as an argument, not a constant.
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (lossValues, grads) = lg(self.network, Array(arrays[0..<8]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[8]
            self.optimizer.update(model: self.network, gradients: clipped)
            self.optimizer.learningRate = outer
            return lossValues + [norm]
        }
        var stepFn: ([MLXArray]) -> [MLXArray] = step
        if compiled {
            switch compiledSteps[maxT] {
            case .some(let cached):
                stepFn = cached
            case .none:
                let fn = compile(inputs: [network, optimizer], outputs: [network, optimizer], step)
                compiledSteps[maxT] = fn
                stepFn = fn
            }
        }
        let lossValues = stepFn([features] + targets.toArrays() + [optimizer.learningRate])
        eval(network, optimizer, lossValues)
        lastGradientNorm = lossValues[1].item(Float.self)
        return lossValues[0].item(Float.self)
    }
}
