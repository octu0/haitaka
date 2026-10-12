import Accelerate
import Foundation

/// CPU inference for the Hawk acoustic model. Processes audio chunk by chunk, carrying the
/// RG-LRU memory and buffered convolution frames across chunks.
/// Linear layers run as batched matrix products over the chunk (Accelerate); gates and the recurrence use SIMD8.
/// `modelDim`, `rnnDim` and `mlpDim` must be multiples of 8.
public final class HawkNetwork: @unchecked Sendable {
    public let weights: HawkWeights
    /// Per layer, c · log σ(Λ), so that log a_t = r_t · logDecay.
    private let logDecay: [[Float]]

    /// Streaming state for one utterance. With lookahead (`convRight`), each layer holds frames until
    /// their right context arrives, so output lags input by `convRight × numLayers` frames.
    public final class State {
        /// Per-layer RG-LRU memory, [rnnDim].
        public var memory: [[Float]]
        /// Per-layer convolution input buffer, [rows, rnnDim]. The first `convWidth - 1` rows are past
        /// frames; the remaining rows are frames not yet emitted.
        public var convBuffer: [[Float]]
        /// Per-layer residual stream for frames waiting on the convolution, [rows, modelDim].
        public var pending: [[Float]]

        init(weights: HawkWeights) {
            memory = [[Float]](repeating: [Float](repeating: 0.0, count: weights.rnnDim), count: weights.numLayers)
            convBuffer = [[Float]](repeating: [Float](repeating: 0.0, count: (weights.convWidth - 1) * weights.rnnDim), count: weights.numLayers)
            pending = [[Float]](repeating: [], count: weights.numLayers)
        }
    }

    public init(weights: HawkWeights) {
        precondition(weights.modelDim % 8 == 0 && weights.rnnDim % 8 == 0 && weights.mlpDim % 8 == 0, "widths must be multiples of 8")
        self.weights = weights
        self.logDecay = weights.layers.map { layer in
            layer.lambda.map { lambda -> Float in
                // Numerically stable log σ(Λ).
                var logSigmoid = -log1pf(expf(-lambda))
                if lambda < 0.0 {
                    logSigmoid = lambda - log1pf(expf(lambda))
                }
                return HawkWeights.decayExponent * logSigmoid
            }
        }
    }

    public func makeState() -> State {
        return State(weights: weights)
    }

    /// Log-probabilities [T][outputDim] for a whole utterance.
    public func logProbs(features: [[Float]], chunkFrames: Int = 128) -> [[Float]] {
        let state = makeState()
        var result: [[Float]] = []
        result.reserveCapacity(features.count)
        var t0 = 0
        while t0 < features.count {
            let t1 = min(features.count, t0 + max(1, chunkFrames))
            result += push(Array(features[t0..<t1]), state: state)
            t0 = t1
        }
        result += finish(state: state)
        return result
    }

    /// Feeds feature frames and returns log-probabilities for the frames whose right context is complete.
    /// With lookahead the output lags the input by `weights.lookaheadFrames`; `finish` drains the rest.
    public func push(_ features: [[Float]], state: State) -> [[Float]] {
        if features.isEmpty {
            return []
        }
        let inDim = weights.inputDim
        var chunk = [Float](repeating: 0.0, count: features.count * inDim)
        var t = 0
        while t < features.count {
            let row = features[t]
            let base = t * inDim
            var d = 0
            while d < min(inDim, row.count) {
                chunk[base + d] = row[d]
                d += 1
            }
            t += 1
        }
        return logSoftmaxRows(forward(chunk: chunk, frames: features.count, state: state))
    }

    /// Ends the utterance: pads every layer's right context with zeros and returns the remaining frames.
    public func finish(state: State) -> [[Float]] {
        let w = weights
        var x: [Float] = []
        var l = 0
        while l < w.numLayers {
            ingest(layer: l, input: x, state: state)
            state.convBuffer[l] += [Float](repeating: 0.0, count: w.convRight * w.rnnDim)
            x = produce(layer: l, state: state)
            l += 1
        }
        return logSoftmaxRows(readout(x))
    }

    private func forward(chunk: [Float], frames: Int, state: State) -> [Float] {
        let w = weights
        let d = w.modelDim
        var x = [Float](repeating: 0.0, count: frames * d)
        var f = 0
        while f < frames {
            var c = 0
            while c < d {
                x[f * d + c] = w.bIn[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(chunk, w.wIn, &x, m: frames, n: d, k: w.inputDim, beta: 1.0)
        var l = 0
        while l < w.numLayers {
            ingest(layer: l, input: x, state: state)
            x = produce(layer: l, state: state)
            l += 1
        }
        return readout(x)
    }

    private func logSoftmaxRows(_ logits: [Float]) -> [[Float]] {
        let outDim = weights.outputDim
        var rows: [[Float]] = []
        rows.reserveCapacity(logits.count / outDim)
        var f = 0
        while f < logits.count / outDim {
            rows.append(Self.logSoftmax(Array(logits[(f * outDim)..<((f + 1) * outDim)])))
            f += 1
        }
        return rows
    }

    /// Feeds frames [rows, modelDim] into a layer and buffers the convolution input (RMSNorm → linear).
    private func ingest(layer l: Int, input: [Float], state: State) {
        let w = weights
        let d = w.modelDim
        let r = w.rnnDim
        let rows = input.count / d
        if rows <= 0 {
            return
        }
        let lw = w.layers[l]
        var normed = [Float](repeating: 0.0, count: rows * d)
        Self.rmsNorm(input, lw.norm1, &normed, frames: rows, width: d)
        var z = [Float](repeating: 0.0, count: rows * r)
        Self.gemm(normed, lw.wX, &z, m: rows, n: r, k: d)
        state.convBuffer[l] += z
        state.pending[l] += input
    }

    /// Runs the rest of the layer on frames whose right context is complete; returns [rows, modelDim].
    private func produce(layer l: Int, state: State) -> [Float] {
        let w = weights
        let d = w.modelDim
        let r = w.rnnDim
        let m = w.mlpDim
        let hd = w.headDim
        let taps = w.convTaps
        let buffered = state.convBuffer[l].count / r
        let frames = buffered - (taps - 1)
        if frames <= 0 {
            return []
        }
        let lw = w.layers[l]
        var x = Array(state.pending[l][0..<(frames * d)])
        state.pending[l].removeFirst(frames * d)

        var normed = [Float](repeating: 0.0, count: frames * d)
        var branch = [Float](repeating: 0.0, count: frames * r)
        var u = [Float](repeating: 0.0, count: frames * r)
        var keepPre = [Float](repeating: 0.0, count: frames * r)
        var admitPre = [Float](repeating: 0.0, count: frames * r)
        var mlpGate = [Float](repeating: 0.0, count: frames * m)
        var mlpUp = [Float](repeating: 0.0, count: frames * m)

        Self.rmsNorm(x, lw.norm1, &normed, frames: frames, width: d)
        Self.gemm(normed, lw.wGate, &branch, m: frames, n: r, k: d)
        Self.geluInPlace(&branch)
        Self.conv(state.convBuffer[l], lw.convW, lw.convB, &u, frames: frames, width: r, taps: taps)
        state.convBuffer[l].removeFirst(frames * r)
        // Block-diagonal gate projections.
        var h = 0
        while h < w.heads {
            let offset = h * hd * hd
            u.withUnsafeBufferPointer { ub in
                lw.wA.withUnsafeBufferPointer { wa in
                    keepPre.withUnsafeMutableBufferPointer { out in
                        Self.gemmRaw(ub.baseAddress! + h * hd, r, wa.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                    }
                }
                lw.wI.withUnsafeBufferPointer { wi in
                    admitPre.withUnsafeMutableBufferPointer { out in
                        Self.gemmRaw(ub.baseAddress! + h * hd, r, wi.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                    }
                }
            }
            h += 1
        }
        scan(layer: l, frames: frames, u: u, keepPre: keepPre, admitPre: admitPre, branch: &branch, memory: &state.memory[l])
        Self.gemm(branch, lw.wO, &x, m: frames, n: d, k: r, beta: 1.0)

        Self.rmsNorm(x, lw.norm2, &normed, frames: frames, width: d)
        Self.gemm(normed, lw.wMLPGate, &mlpGate, m: frames, n: m, k: d)
        Self.gemm(normed, lw.wMLPUp, &mlpUp, m: frames, n: m, k: d)
        Self.geluInPlace(&mlpGate)
        Self.multiplyInPlace(&mlpGate, mlpUp)
        Self.gemm(mlpGate, lw.wMLPDown, &x, m: frames, n: d, k: m, beta: 1.0)
        return x
    }

    /// Final RMSNorm and output projection: [rows, modelDim] → logits [rows, outputDim].
    private func readout(_ x: [Float]) -> [Float] {
        let w = weights
        let d = w.modelDim
        let frames = x.count / d
        if frames <= 0 {
            return []
        }
        var normed = [Float](repeating: 0.0, count: frames * d)
        Self.rmsNorm(x, w.finalNorm, &normed, frames: frames, width: d)
        var logits = [Float](repeating: 0.0, count: frames * w.outputDim)
        var f = 0
        while f < frames {
            var c = 0
            while c < w.outputDim {
                logits[f * w.outputDim + c] = w.bOut[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(normed, w.wOut, &logits, m: frames, n: w.outputDim, k: d, beta: 1.0)
        return logits
    }

    /// RG-LRU: a = exp(r · c · log σ(Λ)), h = a h + √(1 - a²) (i u). Multiplies h into the GeLU branch in place.
    private func scan(layer: Int, frames: Int, u: [Float], keepPre: [Float], admitPre: [Float], branch: inout [Float], memory: inout [Float]) {
        let lw = weights.layers[layer]
        let r = weights.rnnDim
        let one = SIMD8<Float>(repeating: 1.0)
        let eps = SIMD8<Float>(repeating: 1e-6)
        logDecay[layer].withUnsafeBufferPointer { decay in
        lw.bA.withUnsafeBufferPointer { bA in
        lw.bI.withUnsafeBufferPointer { bI in
        u.withUnsafeBufferPointer { ub in
        keepPre.withUnsafeBufferPointer { kp in
        admitPre.withUnsafeBufferPointer { ap in
        branch.withUnsafeMutableBufferPointer { out in
        memory.withUnsafeMutableBufferPointer { mem in
            let dp = decay.baseAddress!
            let bap = bA.baseAddress!
            let bip = bI.baseAddress!
            let up = ub.baseAddress!
            let kpp = kp.baseAddress!
            let app = ap.baseAddress!
            let op = out.baseAddress!
            let mp = mem.baseAddress!
            var c = 0
            while c < r {
                let logA = Self.load(dp + c)
                let keepBias = Self.load(bap + c)
                let admitBias = Self.load(bip + c)
                var state = Self.load(mp + c)
                var t = 0
                while t < frames {
                    let idx = t * r + c
                    let keep = SIMDMath.sigmoid(Self.load(kpp + idx) + keepBias)
                    let admit = SIMDMath.sigmoid(Self.load(app + idx) + admitBias)
                    let a = SIMDMath.exp(keep * logA)
                    let scale = (one - a * a + eps).squareRoot()
                    state = a * state + scale * (admit * Self.load(up + idx))
                    Self.store(state * Self.load(op + idx), op + idx)
                    t += 1
                }
                Self.store(state, mp + c)
                c += 8
            }
        }}}}}}}}
    }

    // MARK: - Helpers

    /// c = a · b (+ beta · c), row-major: a [m, k], b [k, n], c [m, n].
    static func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    gemmRaw(ap.baseAddress!, k, bp.baseAddress!, n, cp.baseAddress!, n, m: m, n: n, k: k, beta: beta)
                }
            }
        }
    }

    static func gemmRaw(_ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int,
                        _ c: UnsafeMutablePointer<Float>, _ ldc: Int, m: Int, n: Int, k: Int, beta: Float) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    1.0, a, Int32(lda), b, Int32(ldb), beta, c, Int32(ldc))
    }

    static func rmsNorm(_ x: [Float], _ gain: [Float], _ out: inout [Float], frames: Int, width: Int) {
        var f = 0
        while f < frames {
            let base = f * width
            var sumSq: Float = 0.0
            var c = 0
            while c < width {
                sumSq += x[base + c] * x[base + c]
                c += 1
            }
            let scale = 1.0 / sqrtf(sumSq / Float(width) + HawkWeights.rmsNormEpsilon)
            c = 0
            while c < width {
                out[base + c] = x[base + c] * scale * gain[c]
                c += 1
            }
            f += 1
        }
    }

    /// Sigmoid approximation of GeLU: x σ(1.702 x).
    static func geluInPlace(_ x: inout [Float]) {
        let k = SIMD8<Float>(repeating: 1.702)
        x.withUnsafeMutableBufferPointer { p in
            let base = p.baseAddress!
            var i = 0
            while i + 8 <= p.count {
                let v = load(base + i)
                store(v * SIMDMath.sigmoid(k * v), base + i)
                i += 8
            }
            while i < p.count {
                p[i] = p[i] / (1.0 + expf(-1.702 * p[i]))
                i += 1
            }
        }
    }

    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func load(_ p: UnsafeMutablePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    static func multiplyInPlace(_ x: inout [Float], _ y: [Float]) {
        var i = 0
        while i < x.count {
            x[i] *= y[i]
            i += 1
        }
    }

    /// Depthwise temporal convolution. `buffer` holds at least [frames + taps - 1, width];
    /// output frame t reads rows t ..< t + taps.
    static func conv(_ buffer: [Float], _ convW: [Float], _ convB: [Float], _ out: inout [Float], frames: Int, width: Int, taps: Int) {
        buffer.withUnsafeBufferPointer { bp in
        convW.withUnsafeBufferPointer { wp in
        convB.withUnsafeBufferPointer { cb in
        out.withUnsafeMutableBufferPointer { op in
            let b = bp.baseAddress!
            let w = wp.baseAddress!
            let bias = cb.baseAddress!
            let o = op.baseAddress!
            var t = 0
            while t < frames {
                var c = 0
                while c < width {
                    var acc = load(bias + c)
                    var j = 0
                    while j < taps {
                        acc += load(w + j * width + c) * load(b + (t + j) * width + c)
                        j += 1
                    }
                    store(acc, o + t * width + c)
                    c += 8
                }
                t += 1
            }
        }}}}
    }

    static func logSoftmax(_ logits: [Float]) -> [Float] {
        var peak = -Float.greatestFiniteMagnitude
        for v in logits {
            if peak < v {
                peak = v
            }
        }
        var total: Float = 0.0
        for v in logits {
            total += expf(v - peak)
        }
        let norm = peak + logf(total)
        return logits.map { $0 - norm }
    }
}
