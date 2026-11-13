import XCTest
@testable import Haitaka

final class FeatureAugmenterTests: XCTestCase {
    func testResampleChangesLengthAndKeepsShape() {
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < pcm.count {
            pcm[i] = sin(Float(i) * 0.01)
            i += 1
        }
        let faster = FeatureAugmenter.resample(pcm, speed: 1.1)
        let slower = FeatureAugmenter.resample(pcm, speed: 0.9)
        XCTAssertEqual(faster.count, Int(16000.0 / 1.1))
        XCTAssertEqual(slower.count, Int(16000.0 / 0.9))
        // At speed 1.1, output sample i is near input sample 1.1 i.
        XCTAssertEqual(faster[1000], sin(1100.0 * 0.01), accuracy: 1e-3)
        XCTAssertEqual(FeatureAugmenter.resample(pcm, speed: 1.0), pcm)
    }

    func testAugmentationIsSeededByPathAndEpoch() {
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < pcm.count {
            pcm[i] = sin(Float(i) * 0.02)
            i += 1
        }
        let augmenter = FeatureAugmenter(speed: true)
        let transform = augmenter.pcmTransform!
        XCTAssertEqual(transform("a.wav", pcm), transform("a.wav", pcm))
        XCTAssertNotEqual(transform("a.wav", pcm).count, transform("b.wav", pcm).count)
        let first = transform("a.wav", pcm)
        augmenter.epochSalt = 2
        XCTAssertNotEqual(transform("a.wav", pcm).count, first.count)
        XCTAssertNotEqual(FeatureAugmenter.seed(forPath: "a.wav"), FeatureAugmenter.seed(forPath: "b.wav"))
    }

    func testRepeatedRowsGetDifferentKeys() {
        XCTAssertNotEqual(FeatureAugmenter.key(path: "a.wav", index: 3), FeatureAugmenter.key(path: "a.wav", index: 7))
        let augmenter = FeatureAugmenter(speed: true)
        var differs = false
        var index = 1
        while index < 20 {
            let first = augmenter.speedFactor(path: FeatureAugmenter.key(path: "a.wav", index: 0))
            if first != augmenter.speedFactor(path: FeatureAugmenter.key(path: "a.wav", index: index)) {
                differs = true
            }
            index += 1
        }
        XCTAssertTrue(differs)
    }

    func testDisabledAugmenterHasNoHooks() {
        let off = FeatureAugmenter(speed: false)
        XCTAssertNil(off.pcmTransform)
        let on = FeatureAugmenter(speed: true)
        XCTAssertNotNil(on.pcmTransform)
    }

    /// Batches are formed on the predicted perturbed length, so it must match the features of the perturbed audio.
    func testPredictedFrameCountMatchesPerturbedFeatures() {
        let augmenter = FeatureAugmenter(speed: true)
        var pcm = [Float](repeating: 0.0, count: 16000 * 3)
        var i = 0
        while i < pcm.count {
            let t = Float(i)
            pcm[i] = 0.1 * sin(t * 0.05) + 0.02 * sin(t * 0.31)
            i += 1
        }
        let frames = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm).count
        var worst = 0
        for epoch in 1...6 {
            augmenter.epochSalt = UInt64(epoch)
            for path in ["/a/x.wav", "/b/y.wav", "/c/z.wav"] {
                let perturbed = SpeechDataset.extractFeaturesFromPCM(pcmData: augmenter.transformPCM(path: path, pcm)).count
                worst = max(worst, abs(augmenter.frameCount(frames, path: path) - perturbed))
            }
        }
        XCTAssertLessThanOrEqual(worst, 1)
        XCTAssertEqual(FeatureAugmenter(speed: false).frameCount(frames, path: "/a/x.wav"), frames)
    }
}
