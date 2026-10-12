import XCTest
@testable import Haitaka

final class SpeechDatasetTests: XCTestCase {
    func testFeaturesHaveModelInputWidth() {
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < pcm.count {
            pcm[i] = 0.1 * sinf(2.0 * Float.pi * 220.0 * Float(i) / 16000.0)
            i += 1
        }
        let features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm)
        XCTAssertEqual(features.count, 24)
        for frame in features {
            XCTAssertEqual(frame.count, StreamingFeatureFrontEnd.inputDim)
        }
    }

    func testResampleFrom48kAveragesTriplets() {
        let pcm: [Float] = [0.0, 0.3, 0.6, 1.0, 1.0, 1.0]
        let out = SpeechDataset.resampleTo16k(pcmData: pcm, sampleRate: 48000)
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(out[1], 1.0, accuracy: 1e-6)
    }
}
