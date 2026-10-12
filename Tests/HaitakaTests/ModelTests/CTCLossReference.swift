import Foundation

/// Reference CPU implementation of the CTC loss and its gradient, used to check `MLXCTCLoss`.
struct CTCLossCalculator: Sendable {
    let blankId: Int

    init(blankId: Int = 0) {
        self.blankId = blankId
    }

    /// log(exp(a) + exp(b))
    @inline(__always)
    static func logAdd(_ a: Float, _ b: Float) -> Float {
        if a == -.infinity { return b }
        if b == -.infinity { return a }
        if a < b {
            return b + log(1.0 + exp(a - b))
        }
        return a + log(1.0 + exp(b - a))
    }

    /// CTC loss and its gradient w.r.t. the pre-softmax logits [T, V], from log-probabilities [T, V] and targets [U].
    func computeLossAndGradients(
        logProbs: [[Float]],       // [T, V]
        targets: [Int]             // [U]
    ) -> (loss: Float, gradients: [[Float]]) {
        let tCount = logProbs.count
        let uCount = targets.count

        if tCount <= 0 || logProbs[0].isEmpty {
            return (loss: 0.0, gradients: [])
        }

        let vCount = logProbs[0].count
        if uCount <= 0 {
            // Empty target: every frame is blank.
            var loss: Float = 0.0
            var grads = [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount)
            var t = 0
            while t < tCount {
                let lp = logProbs[t]
                loss -= lp[blankId]
                var v = 0
                while v < vCount {
                    grads[t][v] = exp(lp[v])
                    v += 1
                }
                grads[t][blankId] -= 1.0
                t += 1
            }
            return (loss: loss, gradients: grads)
        }

        // 1. Extended target l' of length L = 2U + 1.
        let lLen = (uCount * 2) + 1
        var extendedTargets = [Int](repeating: blankId, count: lLen)
        var u = 0
        while u < uCount {
            extendedTargets[(u * 2) + 1] = targets[u]
            u += 1
        }

        if tCount < uCount {
            // Fewer frames than targets: no valid alignment.
            return (loss: Float(uCount * 10), gradients: [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount))
        }

        // 2. Log-space forward variables alpha [T, L].
        var alpha = [[Float]](repeating: [Float](repeating: -.infinity, count: lLen), count: tCount)
        alpha[0][0] = logProbs[0][blankId]
        if 1 < lLen {
            alpha[0][1] = logProbs[0][extendedTargets[1]]
        }

        var t = 1
        while t < tCount {
            let lp = logProbs[t]
            var s = 0
            while s < lLen {
                let targetId = extendedTargets[s]
                let logP = lp[targetId]

                var logSum = alpha[t - 1][s]
                if 0 < s {
                    logSum = Self.logAdd(logSum, alpha[t - 1][s - 1])
                }
                if 1 < s {
                    let prevTarget = extendedTargets[s - 2]
                    if targetId != blankId && targetId != prevTarget {
                        logSum = Self.logAdd(logSum, alpha[t - 1][s - 2])
                    }
                }

                if logSum != -.infinity {
                    alpha[t][s] = logSum + logP
                }
                s += 1
            }
            t += 1
        }

        // log P(Y|X)
        var term2: Float = -.infinity
        if 1 < lLen {
            term2 = alpha[tCount - 1][lLen - 2]
        }
        let logTotalProb = Self.logAdd(alpha[tCount - 1][lLen - 1], term2)
        let loss = -logTotalProb

        if logTotalProb == -.infinity || loss.isNaN {
            return (loss: Float(uCount * 5), gradients: [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount))
        }

        // 3. Log-space backward variables beta [T, L].
        var beta = [[Float]](repeating: [Float](repeating: -.infinity, count: lLen), count: tCount)
        beta[tCount - 1][lLen - 1] = 0.0
        if 1 < lLen {
            beta[tCount - 1][lLen - 2] = 0.0
        }

        t = tCount - 2
        while 0 <= t {
            let nextLp = logProbs[t + 1]
            var s = 0
            while s < lLen {
                var logSum = beta[t + 1][s] + nextLp[extendedTargets[s]]
                if (s + 1) < lLen {
                    logSum = Self.logAdd(logSum, beta[t + 1][s + 1] + nextLp[extendedTargets[s + 1]])
                }
                if (s + 2) < lLen {
                    let currTarget = extendedTargets[s]
                    let nextNextTarget = extendedTargets[s + 2]
                    if currTarget != blankId && currTarget != nextNextTarget {
                        logSum = Self.logAdd(logSum, beta[t + 1][s + 2] + nextLp[nextNextTarget])
                    }
                }

                beta[t][s] = logSum
                s += 1
            }
            t -= 1
        }

        // 4. Gradient: dL/dlogit_{t,k} = exp(logProbs[t][k]) - gamma_t(k).
        var gradients = [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount)
        t = 0
        while t < tCount {
            let lp = logProbs[t]
            // Posterior gamma_t(k) for each class k.
            var logGamma = [Float](repeating: -.infinity, count: vCount)
            var s = 0
            while s < lLen {
                let k = extendedTargets[s]
                let ab = alpha[t][s] + beta[t][s]
                logGamma[k] = Self.logAdd(logGamma[k], ab)
                s += 1
            }

            var k = 0
            while k < vCount {
                let prob = exp(lp[k])
                var postProb: Float = 0.0
                if logGamma[k] != -.infinity {
                    postProb = exp(logGamma[k] - logTotalProb)
                }
                gradients[t][k] = prob - postProb
                k += 1
            }
            t += 1
        }

        return (loss: loss, gradients: gradients)
    }
}
