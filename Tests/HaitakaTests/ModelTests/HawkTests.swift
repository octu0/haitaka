import XCTest
import MLX
@testable import Haitaka

/// Consistency between the MLX training model and the CPU inference network.
/// Runs on the GPU because the linear scan is a Metal kernel.
final class HawkTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .gpu)
        MLXRandom.seed(7)
    }

    private func makeFeatures(frames: Int, dim: Int, phase: Int = 0) -> [[Float]] {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: dim)
            var d = 0
            while d < dim {
                frame[d] = sin(Float((t + phase) * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return seq
    }

    private func smallNetwork(convRight: Int = 0) -> MLXHawkNetwork {
        return MLXHawkNetwork(inputDim: 12, modelDim: 16, rnnDim: 24, mlpDim: 32, heads: 3, convWidth: 4, convRight: convRight, numLayers: 2, outputDim: 7)
    }

    /// Reference scan as an explicit time loop (differentiable through the unrolled graph).
    private func sequentialScan(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        var h = MLXArray.zeros([a.shape[0], a.shape[2]])
        var outs: [MLXArray] = []
        var t = 0
        while t < a.shape[1] {
            h = a[0..., t, 0...] * h + x[0..., t, 0...]
            outs.append(h)
            t += 1
        }
        return stacked(outs, axis: 1)
    }

    private func assertClose(_ a: MLXArray, _ b: MLXArray, _ tolerance: Float, file: StaticString = #filePath, line: UInt = #line) {
        eval(a, b)
        let x = a.asArray(Float.self)
        let y = b.asArray(Float.self)
        XCTAssertEqual(x.count, y.count, file: file, line: line)
        var worst: Float = 0.0
        var i = 0
        while i < x.count {
            worst = max(worst, abs(x[i] - y[i]))
            i += 1
        }
        XCTAssertLessThan(worst, tolerance, file: file, line: line)
    }

    func testLinearScanMatchesSequentialForwardAndGradient() {
        let a = MLXRandom.uniform(low: 0.2, high: 0.99, [3, 41, 10])
        let x = MLXRandom.normal([3, 41, 10])
        let weight = MLXRandom.normal([3, 41, 10])
        assertClose(MLXLinearScan.scan(a, x), sequentialScan(a, x), 1e-4)

        let kernelGrad = grad({ (arrays: [MLXArray]) -> MLXArray in
            sum(MLXLinearScan.scan(arrays[0], arrays[1]) * weight)
        })([a, x])
        let referenceGrad = grad({ (arrays: [MLXArray]) -> MLXArray in
            sum(self.sequentialScan(arrays[0], arrays[1]) * weight)
        })([a, x])
        assertClose(kernelGrad[0], referenceGrad[0], 1e-3)
        assertClose(kernelGrad[1], referenceGrad[1], 1e-3)
    }

    /// Reference local attention as a dense masked softmax over all frame pairs.
    private func denseLocalAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, window: Int) -> MLXArray {
        let frames = q.shape[1]
        let scores = matmul(q.transposed(0, 2, 1, 3), k.expandedDimensions(axis: 1).transposed(0, 1, 3, 2))
        let t = MLXArray(0..<frames).reshaped([frames, 1])
        let s = MLXArray(0..<frames).reshaped([1, frames])
        let allowed = logicalAnd(s .<= t, t - s .< window)
        let p = softmax(MLX.where(allowed, scores, MLXArray(Float(-1e9))), axis: -1)
        return matmul(p, v.expandedDimensions(axis: 1)).transposed(0, 2, 1, 3)
    }

    /// The fused local attention kernel matches the dense reference in value and gradient, for head sizes and
    /// windows that do not fill whole SIMD groups and for windows wider than 32 offsets.
    func testLocalAttentionMatchesDenseForwardAndGradient() {
        for window in [9, 40] {
            let q = MLXRandom.normal([2, 45, 3, 36])
            let k = MLXRandom.normal([2, 45, 36])
            let v = MLXRandom.normal([2, 45, 36])
            let weight = MLXRandom.normal([2, 45, 3, 36])
            assertClose(MLXLocalAttention.attend(q, k, v, window: window), denseLocalAttention(q, k, v, window: window), 1e-4)

            // One input at a time: grad over a three-array list returns zero for the third.
            var i = 0
            while i < 3 {
                func inputs(_ x: MLXArray) -> [MLXArray] {
                    var arrays = [q, k, v]
                    arrays[i] = x
                    return arrays
                }
                let kernelGrad = grad({ (x: MLXArray) -> MLXArray in
                    let a = inputs(x)
                    return sum(MLXLocalAttention.attend(a[0], a[1], a[2], window: window) * weight)
                })([q, k, v][i])
                let referenceGrad = grad({ (x: MLXArray) -> MLXArray in
                    let a = inputs(x)
                    return sum(self.denseLocalAttention(a[0], a[1], a[2], window: window) * weight)
                })([q, k, v][i])
                assertClose(kernelGrad, referenceGrad, 1e-3)
                i += 1
            }
        }
    }

    /// Chunked CPU inference with carried state matches the full-sequence MLX forward pass.
    func testForwardMatchesBetweenMLXAndCPU() {
        assertForwardMatches(convRight: 0, chunkFrames: 5)
    }

    /// With lookahead: delayed CPU output, flushed with zero padding, matches the full-sequence MLX forward pass.
    func testForwardMatchesWithLookahead() {
        assertForwardMatches(convRight: 2, chunkFrames: 5)
        assertForwardMatches(convRight: 1, chunkFrames: 1)
        assertForwardMatches(convRight: 3, chunkFrames: 64)
    }

    private func assertForwardMatches(convRight: Int, chunkFrames: Int) {
        let net = smallNetwork(convRight: convRight)
        // Move trainable biases and norm gains away from their initial values so the test exercises them.
        let start = net.exportWeights()
        let shifted = HawkWeights(
            inputDim: start.inputDim, modelDim: start.modelDim, rnnDim: start.rnnDim, mlpDim: start.mlpDim,
            heads: start.heads, convWidth: start.convWidth, convRight: start.convRight, outputDim: start.outputDim, wIn: start.wIn, bIn: start.bIn,
            layers: start.layers.map { l in
                HawkLayerWeights(
                    norm1: l.norm1.map { $0 * 1.3 }, wGate: l.wGate, wX: l.wX, convW: l.convW.map { $0 * 1.5 }, convB: l.convB.map { _ in 0.1 },
                    wA: l.wA, bA: l.bA.map { _ in 0.4 }, wI: l.wI, bI: l.bI.map { _ in -0.3 }, lambda: l.lambda, wO: l.wO,
                    norm2: l.norm2, wMLPGate: l.wMLPGate, wMLPUp: l.wMLPUp, wMLPDown: l.wMLPDown
                )
            },
            finalNorm: start.finalNorm, wOut: start.wOut, bOut: start.bOut.map { _ in 0.05 }
        )
        net.importWeights(from: shifted)
        let frames = 37
        let features = makeFeatures(frames: frames, dim: 12)
        let logits = net(MLXArray(features.flatMap { $0 }, [1, frames, 12]))
        let expected = logits - logSumExp(logits, axis: -1, keepDims: true)

        let cpu = HawkNetwork(weights: net.exportWeights())
        let got = cpu.logProbs(features: features, chunkFrames: chunkFrames)
        XCTAssertEqual(got.count, frames)
        assertClose(MLXArray(got.flatMap { $0 }, [1, frames, 7]), expected, 2e-4)
    }

    /// Local attention blocks: chunked CPU inference with cached keys matches the MLX forward pass, with and
    /// without lookahead, and for sequences longer than the window.
    func testForwardMatchesWithLocalAttention() {
        for (convRight, chunk) in [(0, 5), (2, 3), (1, 64)] {
            let net = MLXHawkNetwork(inputDim: 12, modelDim: 16, rnnDim: 24, mlpDim: 32, heads: 3, convWidth: 4, convRight: convRight,
                                     numLayers: 2, outputDim: 7, attentionAfter: [0, 1], attentionHeads: 2, attentionWindow: 9)
            let frames = 37
            let features = makeFeatures(frames: frames, dim: 12)
            let logits = net(MLXArray(features.flatMap { $0 }, [1, frames, 12]))
            let expected = logits - logSumExp(logits, axis: -1, keepDims: true)
            let weights = net.exportWeights()
            XCTAssertEqual(weights.attention.count, 2)
            XCTAssertEqual(weights.parameterCount, net.parameters().flattened().reduce(0) { $0 + $1.1.size })
            let got = HawkNetwork(weights: weights).logProbs(features: features, chunkFrames: chunk)
            XCTAssertEqual(got.count, frames)
            assertClose(MLXArray(got.flatMap { $0 }, [1, frames, 7]), expected, 2e-4)
        }
    }

    /// Weights saved without attention still load, with no attention blocks.
    func testWeightsWithoutAttentionStillDecode() throws {
        let weights = smallNetwork(convRight: 1).exportWeights()
        var json = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(weights)) as! [String: Any]
        json.removeValue(forKey: "attention")
        json.removeValue(forKey: "attentionHeads")
        json.removeValue(forKey: "attentionWindow")
        let decoded = try JSONDecoder().decode(HawkWeights.self, from: try JSONSerialization.data(withJSONObject: json))
        XCTAssertEqual(decoded, weights)
        XCTAssertTrue(decoded.attention.isEmpty)
    }

    /// In a padded batch, frames past a sequence's length enter the lookahead convolution as zeros, as in the CPU flush.
    func testFrameCountsMaskMatchesSingleSequence() {
        let net = smallNetwork(convRight: 2)
        let short = makeFeatures(frames: 20, dim: 12)
        let long = makeFeatures(frames: 33, dim: 12, phase: 3)
        var padded = short
        padded += [[Float]](repeating: [Float](repeating: 0.0, count: 12), count: 13)
        let batch = MLXArray((padded + long).flatMap { $0 }, [2, 33, 12])
        let batched = net(batch, frameCounts: MLXArray([Int32(20), Int32(33)]))
        let single = net(MLXArray(short.flatMap { $0 }, [1, 20, 12]))
        assertClose(batched[0, 0..<20, 0...], single[0], 1e-4)
    }

    func testWeightsRoundTripThroughFile() throws {
        let net = smallNetwork(convRight: 1)
        let weights = net.exportWeights()
        XCTAssertEqual(weights.parameterCount, net.parameters().flattened().reduce(0) { $0 + $1.1.size })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hawk_roundtrip_\(UUID().uuidString).json")
        try weights.save(to: url)
        let loaded = try HawkWeights.load(from: url)
        XCTAssertEqual(loaded, weights)
        let other = MLXHawkNetwork(weights: loaded)
        XCTAssertEqual(other.exportWeights(), weights)
    }

    /// Gradients through 8 layers stay finite on long runs of all-zero (silent) feature frames.
    func testGradientStaysFiniteThroughSilentFrames() {
        let net = MLXHawkNetwork(inputDim: 12, modelDim: 32, rnnDim: 32, mlpDim: 48, heads: 4, convWidth: 4, convRight: 0, numLayers: 8, outputDim: 7)
        var features = makeFeatures(frames: 60, dim: 12)
        var t = 20
        while t < 40 {
            features[t] = [Float](repeating: 0.0, count: 12)
            t += 1
        }
        let input = MLXArray(features.flatMap { $0 }, [1, 60, 12])
        let g = grad({ (f: MLXArray) -> MLXArray in
            let logits = net(f)
            return sum(logits * logits)
        })(input)
        eval(g)
        XCTAssertFalse(any(isNaN(g) .|| isInf(g)).item(Bool.self))
        XCTAssertLessThan(abs(g).max().item(Float.self), 1e6)
    }

    func testCTCTrainingReducesLoss() {
        let net = smallNetwork()
        let before = net.exportWeights()
        let trainer = MLXHawkTrainer(network: net, learningRate: 0.01)
        let batch: [[[Float]]] = [makeFeatures(frames: 24, dim: 12), makeFeatures(frames: 20, dim: 12, phase: 5)]
        let targets: [[Int]] = [[1, 2, 3, 2], [4, 5, 6]]
        let first = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets)
        var last = first
        var step = 0
        while step < 40 {
            last = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets)
            step += 1
        }
        XCTAssertFalse(last.isNaN)
        XCTAssertLessThan(last, first * 0.5)
        let after = net.exportWeights()
        XCTAssertNotEqual(after.layers[1].lambda, before.layers[1].lambda)
        XCTAssertNotEqual(after.layers[0].wA, before.layers[0].wA)
    }
}
