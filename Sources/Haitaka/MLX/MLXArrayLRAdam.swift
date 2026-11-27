import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Adam first and second moments.
public struct ArrayLRAdamState: Updatable {
    public let m: MLXArray
    public let v: MLXArray

    public func innerState() -> [MLXArray] {
        return [m, v]
    }
}

/// Adam without bias correction whose learning rate is an `MLXArray`, with optional decoupled weight decay.
///
/// MLX's `Adam` stores the learning rate as a `Float`, which `compile()` bakes into the traced graph,
/// so a learning-rate schedule would have no effect on a cached compiled step. Passing it as an array
/// input avoids that. Implements `Optimizer` directly because `OptimizerBase` has an internal initializer.
public final class ArrayLRAdam: Optimizer {
    public var learningRate: MLXArray
    public let betas: (Float, Float)
    public let eps: Float
    /// Decoupled weight decay (AdamW), applied only to weight matrices: tensors of 2 or more dimensions, except small
    /// 2-D kernels whose first dimension is at most `decayMinRows` (depthwise convolution taps). 1-D tensors (RG-LRU
    /// `lambda`, norm gains, biases) are never decayed, since pulling `lambda` to zero would collapse the memory.
    public let weightDecay: Float
    public static let decayMinRows = 8

    private var stateStorage = NestedDictionary<String, ArrayLRAdamState>()

    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8, weightDecay: Float = 0.0) {
        self.learningRate = MLXArray(learningRate)
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
    }

    /// Whether weight decay applies to a parameter of this shape.
    public static func isDecayed(shape: [Int]) -> Bool {
        if shape.count < 2 {
            return false
        }
        if shape.count == 2 && shape[0] <= decayMinRows {
            return false
        }
        return true
    }

    public func innerState() -> [MLXArray] {
        return stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    /// Zeroes the moments, keeping their shapes. Call after rolling back weights to discard stale gradient history.
    public func resetState() {
        stateStorage = stateStorage.mapValues { state in
            ArrayLRAdamState(m: MLXArray.zeros(like: state.m), v: MLXArray.zeros(like: state.v))
        }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let (b1, b2) = betas
        let lr = learningRate
        let epsValue = eps
        let wd = weightDecay
        let (newParameters, newStates) = gradients.mapValues(model.parameters(), stateStorage) { (gradient: MLXArray, parameter: MLXArray?, state: ArrayLRAdamState?) -> (MLXArray, ArrayLRAdamState?) in
            let p = parameter!
            let mPrev: MLXArray
            let vPrev: MLXArray
            switch state {
            case .some(let s):
                mPrev = s.m
                vPrev = s.v
            case .none:
                mPrev = MLXArray.zeros(like: p)
                vPrev = MLXArray.zeros(like: p)
            }
            let m = b1 * mPrev + (1 - b1) * gradient
            let v = b2 * vPrev + (1 - b2) * square(gradient)
            let update = lr * m / (sqrt(v) + epsValue)
            var decayed = p
            if 0.0 < wd && Self.isDecayed(shape: p.shape) {
                decayed = p * (1.0 - lr * wd)
            }
            return (decayed - update, ArrayLRAdamState(m: m, v: v))
        }
        stateStorage = newStates
        model.update(parameters: newParameters)
    }
}
