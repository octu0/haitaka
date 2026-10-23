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
    private var compiledClusterSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// MLX buffer cache limit at the default width. Each padded sequence length needs its own buffers, so the cache
    /// grows without bound; a limit below one step's working set forces reallocation every step and slows training.
    /// Wider models get a proportionally smaller cache (down to `minCacheLimitBytes`) so cache plus activations fit.
    public static let cacheLimitBytes = 24 << 30
    public static let minCacheLimitBytes = 6 << 30

    public init(network: MLXHawkNetwork, learningRate: Float) {
        self.network = network
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        let scaled = Self.cacheLimitBytes / max(1, network.modelDim / HawkShape.defaultModelDim)
        MLX.GPU.set(cacheLimit: max(Self.minCacheLimitBytes, scaled))
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// Restores saved weights and resets the Adam moments.
    public func rollback(to weights: HawkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        compiledSteps.removeAll()
        compiledClusterSteps.removeAll()
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
        var b = 0
        while b < featuresBatch.count {
            let frames = featuresBatch[b].count
            if 0 < frames && targetsBatch[b].isEmpty != true {
                validFeatures.append(featuresBatch[b])
                validTargets.append(targetsBatch[b])
                frameCounts.append(frames)
            }
            b += 1
        }
        if validFeatures.isEmpty {
            return 0.0
        }
        let (features, maxT) = padded(validFeatures)
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

    /// Runs one HuBERT-style step: spans of input frames are zeroed and the network predicts the cluster class of
    /// each hidden frame. Classes come from the unmasked features, or from layer `teacherLayers` of `teacher`
    /// for the second iteration. Returns the cross-entropy over masked frames.
    public func trainBatchMaskedCluster(
        featuresBatch: [[[Float]]],
        codebook: [MLXArray],
        maskStartProbability: Double,
        maskSpan: Int,
        teacher: MLXHawkNetwork? = nil,
        teacherLayers: Int = 0
    ) -> Float {
        let valid = featuresBatch.filter { $0.isEmpty != true }
        if valid.isEmpty {
            return 0.0
        }
        let (features, maxT) = padded(valid)
        var frameCounts: [Int32] = []
        var span = [Float](repeating: 0.0, count: valid.count * maxT)
        var b = 0
        while b < valid.count {
            let frames = valid[b].count
            frameCounts.append(Int32(frames))
            var t = 0
            while t < frames {
                if Double.random(in: 0.0..<1.0) < maskStartProbability {
                    var k = 0
                    while k < maskSpan && (t + k) < frames {
                        span[b * maxT + t + k] = 1.0
                        k += 1
                    }
                }
                t += 1
            }
            b += 1
        }
        let counts = MLXArray(frameCounts)
        let spanMask = MLXArray(span, [valid.count, maxT])
        var source = features
        if let teacher = teacher {
            source = teacher.hidden(features, frameCounts: counts, layerCount: teacherLayers)
        }
        let targets = ClusterCodebook.assign(features: source, codebook: codebook)
        eval(targets)

        let lossFn = { (model: MLXHawkNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let mask = arrays[1]
            let logits = model(arrays[0] * (1.0 - mask).expandedDimensions(axis: -1), frameCounts: arrays[3])
            let picked = takeAlong(logSoftmax(logits, axis: -1), arrays[2].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            return [-sum(picked * mask) / maximum(sum(mask), MLXArray(Float(1.0)))]
        }
        let lg = valueAndGrad(model: network, lossFn)
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (lossValues, grads) = lg(self.network, Array(arrays[0..<4]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[4]
            self.optimizer.update(model: self.network, gradients: clipped)
            self.optimizer.learningRate = outer
            return lossValues + [norm]
        }
        var stepFn: ([MLXArray]) -> [MLXArray] = step
        switch compiledClusterSteps[maxT] {
        case .some(let cached):
            stepFn = cached
        case .none:
            let fn = compile(inputs: [network, optimizer], outputs: [network, optimizer], step)
            compiledClusterSteps[maxT] = fn
            stepFn = fn
        }
        let lossValues = stepFn([features, spanMask, targets, counts, optimizer.learningRate])
        eval(network, optimizer, lossValues)
        lastGradientNorm = lossValues[1].item(Float.self)
        return lossValues[0].item(Float.self)
    }

    /// Stacks sequences into [B, T, inputDim], zero-padding T up to a multiple of 32 to limit distinct compiled shapes.
    private func padded(_ batch: [[[Float]]]) -> (MLXArray, Int) {
        var maxT = 0
        for f in batch {
            maxT = max(maxT, f.count)
        }
        maxT = ((maxT + 31) / 32) * 32
        let inDim = network.inputDim
        var flat = [Float](repeating: 0.0, count: batch.count * maxT * inDim)
        var b = 0
        while b < batch.count {
            var t = 0
            while t < batch[b].count {
                let offset = (b * maxT + t) * inDim
                let row = batch[b][t]
                var d = 0
                while d < inDim {
                    flat[offset + d] = row[d]
                    d += 1
                }
                t += 1
            }
            b += 1
        }
        return (MLXArray(flat, [batch.count, maxT, inDim]), maxT)
    }
}
