import XCTest
@testable import Haitaka

final class FeatureAugmenterTests: XCTestCase {
    /// A tone over the first half and silence after it, so the noise bank always has room to add noise.
    private func toneThenSilence() -> [Float] {
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < pcm.count / 2 {
            pcm[i] = 0.3 * sin(Float(i) * 0.02)
            i += 1
        }
        return pcm
    }

    private func alwaysNoisy() -> FeatureAugmenter {
        var rng = SplitMix64(seed: 3)
        let noise = (0..<8000).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
        return FeatureAugmenter(noiseBank: NoiseBank(clips: [noise], probability: 1.0, recordedProbability: 1.0))
    }

    func testAugmentationIsSeededByKeyAndEpoch() {
        let pcm = toneThenSilence()
        let augmenter = alwaysNoisy()
        let transform = augmenter.pcmTransform!
        XCTAssertEqual(transform("a.wav", pcm), transform("a.wav", pcm))
        XCTAssertNotEqual(transform("a.wav", pcm), transform("b.wav", pcm))
        let first = transform("a.wav", pcm)
        augmenter.epochSalt = 2
        XCTAssertNotEqual(transform("a.wav", pcm), first)
        XCTAssertNotEqual(FeatureAugmenter.seed(forPath: "a.wav"), FeatureAugmenter.seed(forPath: "b.wav"))
    }

    func testRepeatedRowsGetDifferentKeys() {
        let first = FeatureAugmenter.key(path: "a.wav", index: 3)
        let second = FeatureAugmenter.key(path: "a.wav", index: 7)
        XCTAssertNotEqual(first, second)
        let pcm = toneThenSilence()
        let transform = alwaysNoisy().pcmTransform!
        XCTAssertNotEqual(transform(first, pcm), transform(second, pcm))
    }

    func testAugmenterWithoutBanksHasNoHook() {
        XCTAssertNil(FeatureAugmenter().pcmTransform)
        XCTAssertNotNil(alwaysNoisy().pcmTransform)
    }
}
