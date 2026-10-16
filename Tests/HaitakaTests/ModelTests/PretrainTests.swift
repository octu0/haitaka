import XCTest
import MLX
@testable import Haitaka

/// k-means targets, masked-cluster training, and handing pretrained layers to CTC training.
final class PretrainTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .gpu)
        MLXRandom.seed(11)
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

    private func smallNetwork(outputDim: Int) -> MLXHawkNetwork {
        return MLXHawkNetwork(inputDim: 12, modelDim: 16, rnnDim: 24, mlpDim: 32, heads: 3, convWidth: 4, convRight: 1, numLayers: 2, outputDim: outputDim)
    }

    func testKMeansSeparatesTwoGroups() {
        var frames: [[Float]] = []
        var i = 0
        while i < 40 {
            let offset: Float = Float(i % 2) * 10.0
            frames.append([offset + Float(i % 5) * 0.1, offset - Float(i % 3) * 0.1])
            i += 1
        }
        let book = ClusterCodebook.fit(frames: frames, k: 2, iterations: 10)
        let ids = ClusterCodebook.assign(features: MLXArray(frames.flatMap { $0 }, [frames.count, 2]), codebook: book.arrays())
        eval(ids)
        let labels = ids.asArray(Int32.self)
        XCTAssertNotEqual(labels[0], labels[1])
        var k = 2
        while k < labels.count {
            XCTAssertEqual(labels[k], labels[k % 2])
            k += 1
        }
    }

    func testMaskedClusterTrainingReducesLoss() {
        let batch: [[[Float]]] = [makeFeatures(frames: 40, dim: 12), makeFeatures(frames: 33, dim: 12, phase: 3)]
        let book = ClusterCodebook.fit(frames: batch.flatMap { $0 }, k: 4, iterations: 10)
        let trainer = MLXHawkTrainer(network: smallNetwork(outputDim: 4), learningRate: 0.01)
        // Masks are redrawn every step, so compare averages over several steps.
        var losses: [Float] = []
        while losses.count < 200 {
            losses.append(trainer.trainBatchMaskedCluster(featuresBatch: batch, codebook: book.arrays(), maskStartProbability: 0.3, maskSpan: 3))
        }
        let first = losses.prefix(10).reduce(0, +) / 10.0
        let last = losses.suffix(10).reduce(0, +) / 10.0
        XCTAssertTrue(last.isFinite)
        XCTAssertLessThan(last, first * 0.8)
    }

    func testTeacherTargetsUseHiddenLayer() {
        let teacher = smallNetwork(outputDim: 4)
        let feats = makeFeatures(frames: 32, dim: 12)
        let h = teacher.hidden(MLXArray(feats.flatMap { $0 }, [1, 32, 12]), layerCount: 1)
        eval(h)
        XCTAssertEqual(h.shape, [1, 32, 16])
        let rows = h.asArray(Float.self)
        var frames: [[Float]] = []
        var t = 0
        while t < 32 {
            frames.append(Array(rows[(t * 16)..<((t + 1) * 16)]))
            t += 1
        }
        let book = ClusterCodebook.fit(frames: frames, k: 3, iterations: 5)
        let trainer = MLXHawkTrainer(network: smallNetwork(outputDim: 3), learningRate: 0.01)
        let loss = trainer.trainBatchMaskedCluster(featuresBatch: [feats], codebook: book.arrays(), maskStartProbability: 0.3, maskSpan: 3,
                                                   teacher: teacher, teacherLayers: 1)
        XCTAssertTrue(loss.isFinite)
    }

    func testImportEncoderKeepsReadout() {
        let pretrained = smallNetwork(outputDim: 4).exportWeights()
        let net = smallNetwork(outputDim: 7)
        let readout = net.exportWeights().wOut
        net.importEncoder(from: pretrained)
        let after = net.exportWeights()
        XCTAssertEqual(after.wOut, readout)
        XCTAssertEqual(after.wIn, pretrained.wIn)
        XCTAssertEqual(after.layers[1].lambda, pretrained.layers[1].lambda)
    }
}
