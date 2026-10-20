import Foundation

/// Per-hop feature extraction that builds the acoustic model's input vectors.
///
/// Offline: call `setGain` with the clip RMS, then push hops. Streaming: gain follows the
/// running RMS since speech onset (future samples are not available). The Hamming window is
/// symmetric over the 320-sample frame (changed 2026-10-18; weights trained before that used the
/// first 320 points of a 1024-point table and are not compatible).
public final class StreamingFeatureFrontEnd: @unchecked Sendable {
    public static let melChannels = 64
    /// 64 smoothed mel + 64 deltas
    public static let tapDim = 128
    /// Four 10 ms hops stacked into one 40 ms frame. 10 ms is too fine for CTC, and the number of recurrent steps drives training time.
    public static let frameStack = 4
    /// Target RMS for gain normalization (matches the level of the training data).
    public static let targetRMS: Float = 0.05
    /// Gain cap so near-silent clips do not amplify noise.
    public static let maxGain: Float = 20.0
    /// Acoustic model input dimension (tapDim x frameStack).
    public static let inputDim = tapDim * frameStack

    @inline(__always)
    public static func gainForRMS(_ rms: Float) -> Float {
        if 1e-6 < rms {
            let g = targetRMS / rms
            if maxGain < g {
                return maxGain
            }
            return g
        }
        return 1.0
    }

    public let hopSize: Int
    public let frameSize: Int

    private let filterbank: Filterbank
    private let workspace: DSPWorkspace
    private let preemphasisCoeff: Float

    private var prevMel: [Float]
    private var currMel: [Float]
    private var tapFrame: [Float]
    private var stackBuf: [Float]
    private var preemphBuf: [Float]

    private var hasPrevMel: Bool = false
    private var hasCurrMel: Bool = false
    private var stackFill: Int = 0
    private var emittedStackCount: Int = 0
    private var utteranceFirstFrame: Bool = true
    private var streamRawPrev: Float = 0.0
    private var gain: Float = 1.0
    private var gainFrozen: Bool = false
    private var rmsSumSquares: Float = 0.0
    private var rmsSampleCount: Int = 0

    public init(dspConfig: DSPConfig = DSPConfig()) {
        self.hopSize = dspConfig.hopSize
        self.frameSize = dspConfig.frameSize
        self.preemphasisCoeff = dspConfig.preemphasisCoeff
        self.filterbank = Filterbank(config: dspConfig)
        self.workspace = DSPWorkspace(
            lpcOrder: dspConfig.lpcOrder,
            melChannels: Self.melChannels,
            maxPitchLag: dspConfig.maxPitchLag,
            frameSize: dspConfig.frameSize
        )
        self.prevMel = [Float](repeating: 0.0, count: Self.melChannels)
        self.currMel = [Float](repeating: 0.0, count: Self.melChannels)
        self.tapFrame = [Float](repeating: 0.0, count: Self.tapDim)
        self.stackBuf = [Float](repeating: 0.0, count: Self.inputDim)
        self.preemphBuf = [Float](repeating: 0.0, count: dspConfig.frameSize)
    }

    /// Offline use: sets the gain from a known clip RMS and stops running-RMS updates.
    public func setGain(_ g: Float) {
        gain = g
        gainFrozen = true
    }

    /// Utterance boundary: drops the 3-tap and stacking state. A gain set by `setGain` is kept.
    public func beginUtterance() {
        hasPrevMel = false
        hasCurrMel = false
        stackFill = 0
        emittedStackCount = 0
        utteranceFirstFrame = true
        streamRawPrev = 0.0
        if gainFrozen != true {
            gain = 1.0
            rmsSumSquares = 0.0
            rmsSampleCount = 0
        }
        var i = 0
        while i < stackBuf.count {
            stackBuf[i] = 0.0
            i += 1
        }
    }

    /// Pushes one hop. Returns nil until `frameStack` hops are stacked. The returned buffer is overwritten by the next push / flush.
    @inline(__always)
    public func pushRawFrame(pcmPtr: UnsafePointer<Float>, count: Int) -> [Float]? {
        if count < frameSize {
            return nil
        }
        updateGain(pcmPtr: pcmPtr, count: count)
        applyPreemphasis(pcmPtr: pcmPtr, count: count)
        if ingestMelFromPreemph() != true {
            return nil
        }
        return pushTapIntoStack()
    }

    /// Finalizes the last 3-tap with next = curr. A pending stack is zero-padded into one frame; any other remainder is dropped.
    public func flush() -> [Float]? {
        if hasCurrMel != true {
            return nil
        }
        if hasPrevMel != true {
            writeThreeTap(prev: currMel, curr: currMel, next: currMel)
        } else {
            writeThreeTap(prev: prevMel, curr: currMel, next: currMel)
        }
        hasCurrMel = false
        hasPrevMel = false
        if let stacked = pushTapIntoStack() {
            return stacked
        }
        if 0 < stackFill && emittedStackCount == 0 {
            var i = stackFill * Self.tapDim
            while i < Self.inputDim {
                stackBuf[i] = 0.0
                i += 1
            }
            stackFill = 0
            emittedStackCount += 1
            return stackBuf
        }
        return nil
    }

    @inline(__always)
    private func updateGain(pcmPtr: UnsafePointer<Float>, count: Int) {
        if gainFrozen {
            return
        }
        if utteranceFirstFrame {
            var i = 0
            while i < count {
                let x = pcmPtr[i]
                rmsSumSquares += x * x
                i += 1
            }
            rmsSampleCount += count
        } else {
            var addCount = hopSize
            if count < hopSize {
                addCount = count
            }
            let start = count - addCount
            var src = start
            while src < count {
                let x = pcmPtr[src]
                rmsSumSquares += x * x
                src += 1
            }
            rmsSampleCount += addCount
        }
        var rms: Float = 0.0
        if 0 < rmsSampleCount {
            rms = sqrtf(rmsSumSquares / Float(rmsSampleCount))
        }
        gain = Self.gainForRMS(rms)
    }

    @inline(__always)
    private func applyPreemphasis(pcmPtr: UnsafePointer<Float>, count: Int) {
        let coeff = preemphasisCoeff
        if utteranceFirstFrame {
            preemphBuf[0] = pcmPtr[0] * gain
            utteranceFirstFrame = false
        } else {
            preemphBuf[0] = (pcmPtr[0] - (coeff * streamRawPrev)) * gain
        }
        var i = 1
        while i < count {
            preemphBuf[i] = (pcmPtr[i] - (coeff * pcmPtr[i - 1])) * gain
            i += 1
        }
        streamRawPrev = pcmPtr[hopSize - 1]
    }

    @inline(__always)
    private func ingestMelFromPreemph() -> Bool {
        let mel = preemphBuf.withUnsafeBufferPointer { buf in
            return filterbank.extractFeatures(
                pcmPtr: buf.baseAddress!,
                count: frameSize,
                workspace: workspace
            )
        }
        if hasCurrMel != true {
            currMel = mel
            hasCurrMel = true
            return false
        }
        if hasPrevMel != true {
            writeThreeTap(prev: currMel, curr: currMel, next: mel)
            prevMel = currMel
            currMel = mel
            hasPrevMel = true
            return true
        }
        writeThreeTap(prev: prevMel, curr: currMel, next: mel)
        prevMel = currMel
        currMel = mel
        return true
    }

    @inline(__always)
    private func writeThreeTap(prev: [Float], curr: [Float], next: [Float]) {
        var c = 0
        while c < Self.melChannels {
            let xPrev = prev[c]
            let xCurr = curr[c]
            let xNext = next[c]
            tapFrame[c] = (0.25 * xPrev) + (0.5 * xCurr) + (0.25 * xNext)
            tapFrame[Self.melChannels + c] = 0.5 * (xNext - xPrev)
            c += 1
        }
    }

    @inline(__always)
    private func pushTapIntoStack() -> [Float]? {
        let dim = Self.tapDim
        let offset = stackFill * dim
        var d = 0
        while d < dim {
            stackBuf[offset + d] = tapFrame[d]
            d += 1
        }
        stackFill += 1
        if stackFill < Self.frameStack {
            return nil
        }
        stackFill = 0
        emittedStackCount += 1
        return stackBuf
    }
}
