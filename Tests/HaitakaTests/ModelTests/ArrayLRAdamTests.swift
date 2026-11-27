import XCTest
import MLX
import MLXNN
@testable import Haitaka

final class ArrayLRAdamTests: XCTestCase {
    final class Tiny: Module {
        var matrix: MLXArray
        var taps: MLXArray
        var gain: MLXArray

        override init() {
            matrix = MLXArray.ones([16, 4])
            taps = MLXArray.ones([6, 4])
            gain = MLXArray.ones([4])
            super.init()
        }
    }

    func testOnlyWeightMatricesAreDecayed() {
        XCTAssertTrue(ArrayLRAdam.isDecayed(shape: [16, 4]))
        XCTAssertTrue(ArrayLRAdam.isDecayed(shape: [4, 3, 3]))
        XCTAssertFalse(ArrayLRAdam.isDecayed(shape: [6, 4]), "depthwise convolution taps")
        XCTAssertFalse(ArrayLRAdam.isDecayed(shape: [4]), "lambda, norm gains and biases")
    }

    func testDecayShrinksMatricesAndLeavesTheRest() {
        let model = Tiny()
        let optimizer = ArrayLRAdam(learningRate: 0.1, weightDecay: 0.5)
        let zeros = model.parameters().mapValues { MLXArray.zeros(like: $0) }
        optimizer.update(model: model, gradients: zeros)
        eval(model)
        XCTAssertEqual(model.matrix.max().item(Float.self), 0.95, accuracy: 1e-5)
        XCTAssertEqual(model.taps.max().item(Float.self), 1.0, accuracy: 1e-6)
        XCTAssertEqual(model.gain.max().item(Float.self), 1.0, accuracy: 1e-6)
    }

    func testNoDecayByDefault() {
        let model = Tiny()
        let optimizer = ArrayLRAdam(learningRate: 0.1)
        let zeros = model.parameters().mapValues { MLXArray.zeros(like: $0) }
        optimizer.update(model: model, gradients: zeros)
        eval(model)
        XCTAssertEqual(model.matrix.max().item(Float.self), 1.0, accuracy: 1e-6)
    }
}
