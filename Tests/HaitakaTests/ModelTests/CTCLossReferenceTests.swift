import Foundation
import XCTest
@testable import Haitaka

final class CTCLossReferenceTests: XCTestCase {

    /// The reference CTC loss is finite and non-negative.
    func testCTCLossComputationAndGradients() {
        let calc = CTCLossCalculator(blankId: 0)
        let vCount = 10
        let tCount = 20

        // Uniform log-probabilities.
        let logP = -log(Float(vCount))
        let logProbs = [[Float]](repeating: [Float](repeating: logP, count: vCount), count: tCount)
        let targets = [1, 2, 3, 4]

        let result = calc.computeLossAndGradients(logProbs: logProbs, targets: targets)
        XCTAssertFalse(result.loss.isNaN, "CTC Loss should not be NaN")
        XCTAssertFalse(result.loss.isInfinite, "CTC Loss should not be infinite")
        XCTAssertLessThan(0.0, result.loss, "CTC Loss should be strictly positive")
        XCTAssertEqual(result.gradients.count, tCount)
        XCTAssertEqual(result.gradients[0].count, vCount)

        // Softmax gradients sum to zero within each frame.
        for t in 0..<tCount {
            var sumGrad: Float = 0.0
            for g in result.gradients[t] {
                sumGrad += g
            }
            XCTAssertLessThanOrEqual(abs(sumGrad), 1e-4, "Softmax gradient sum must be close to 0")
        }
    }
}
