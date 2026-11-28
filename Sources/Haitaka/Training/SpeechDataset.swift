import Foundation

/// Speech/text training set. Only paths and transcripts are kept in memory; features are computed from the
/// WAV file whenever a batch needs them.
public final class SpeechDataset: @unchecked Sendable {
    public struct SampleMeta: Sendable {
        public let path: String
        public let rawText: String
        /// Pronunciation kana used as the CTC target.
        public let hiraganaText: String
        public let frameCount: Int

        public init(path: String, rawText: String, hiraganaText: String, frameCount: Int) {
            self.path = (path as NSString).standardizingPath
            self.rawText = rawText
            self.hiraganaText = hiraganaText
            self.frameCount = frameCount
        }
    }

    public let metaSamples: [SampleMeta]

    public init(metaSamples: [SampleMeta]) {
        self.metaSamples = metaSamples
    }

    public var count: Int {
        return metaSamples.count
    }

    public func frameCount(at index: Int) -> Int {
        return metaSamples[index].frameCount
    }

    public func hiraganaText(at index: Int) -> String {
        return metaSamples[index].hiraganaText
    }

    public func features(at index: Int) -> [[Float]] {
        return Self.loadFeatures(path: metaSamples[index].path).features
    }

    /// Reads a WAV file without loading the whole file into a single `Data`.
    public static func loadWavFile(path: String) -> WavData? {
        do {
            let reader = try WavStreamReader(filePath: path)
            defer { try? reader.close() }
            return try reader.readToEnd()
        } catch {
            return nil
        }
    }

    /// `pcmTransform` runs on 16 kHz PCM before feature extraction (speed perturbation, noise mixing). It is
    /// training-only and receives `key` (the path unless given), which seeds the augmentation.
    public static func loadFeatures(
        path: String,
        key: String? = nil,
        pcmTransform: ((String, [Float]) -> [Float])? = nil
    ) -> (pcm: [Float], features: [[Float]]) {
        guard let wav = loadWavFile(path: path) else {
            return ([], [])
        }
        var pcm16k = resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
        if let transform = pcmTransform {
            pcm16k = transform(key ?? path, pcm16k)
        }
        return (pcm16k, extractFeaturesFromPCM(pcmData: pcm16k))
    }

    /// Builds the dataset from manifest pairs. Utterances that yield no feature frames are dropped.
    public static func lazyFromManifest(
        pairs: [(path: String, text: String)],
        workers: Int = 8,
        english: EnglishPronunciations? = nil
    ) -> SpeechDataset {
        final class MetaBuffer: @unchecked Sendable {
            var items: [SampleMeta?]
            init(count: Int) {
                self.items = [SampleMeta?](repeating: nil, count: count)
            }
        }
        let buffer = MetaBuffer(count: pairs.count)
        let workerCount = max(1, workers)

        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            let converter = KanjiConverter(english: english)
            var i = worker
            while i < pairs.count {
                let pair = pairs[i]
                autoreleasepool {
                    let frameCount = loadFeatures(path: pair.path).features.count
                    if 0 < frameCount {
                        buffer.items[i] = SampleMeta(
                            path: pair.path,
                            rawText: pair.text,
                            hiraganaText: converter.convertToHiragana(pair.text),
                            frameCount: frameCount
                        )
                    }
                }
                i += workerCount
            }
        }
        return SpeechDataset(metaSamples: buffer.items.compactMap { $0 })
    }

    /// Zero crossings on each side of the resampling kernel; the Blackman window stops the band at about -58 dB.
    static let sincHalfCrossings = 16
    /// Kernel table resolution: entries per zero crossing.
    static let sincStepsPerCrossing = 512
    /// Low-pass cutoff relative to the 8 kHz output Nyquist frequency (7.5 kHz).
    static let sincCutoffRatio: Float = 0.9375
    /// g(u) = sinc(u) · Blackman(u / K) for u in [0, K], sampled `sincStepsPerCrossing` times per zero crossing.
    static let sincTable: [Float] = {
        let total = sincHalfCrossings * sincStepsPerCrossing
        var table = [Float](repeating: 0.0, count: total + 1)
        table[0] = 1.0
        var i = 1
        while i <= total {
            let u = Float(i) / Float(sincStepsPerCrossing)
            let piU = Float.pi * u
            let z = u / Float(sincHalfCrossings)
            let window = 0.42 + 0.5 * cosf(Float.pi * z) + 0.08 * cosf(2.0 * Float.pi * z)
            table[i] = sinf(piU) / piU * window
            i += 1
        }
        return table
    }()

    /// Resamples to 16 kHz with a Blackman-windowed sinc low-pass (7.5 kHz cutoff), so content above the output
    /// Nyquist frequency is removed instead of aliasing into the speech band. Each output sample is normalized by its
    /// kernel weights, so the DC gain is exactly 1. 16 kHz input is returned unchanged.
    public static func resampleTo16k(pcmData: [Float], sampleRate: Int) -> [Float] {
        if sampleRate == 16000 || pcmData.isEmpty {
            return pcmData
        }
        let ratio = Float(sampleRate) / 16000.0
        let outCount = Int(Float(pcmData.count) / ratio)
        if outCount <= 0 {
            return []
        }
        let cutoff = sincCutoffRatio / max(1.0, ratio)
        let radius = Float(sincHalfCrossings) / cutoff
        let tableScale = cutoff * Float(sincStepsPerCrossing)
        let tableLimit = sincHalfCrossings * sincStepsPerCrossing
        let inCount = pcmData.count
        var out = [Float](repeating: 0.0, count: outCount)
        pcmData.withUnsafeBufferPointer { srcBuf in
            sincTable.withUnsafeBufferPointer { tblBuf in
                let src = srcBuf.baseAddress!
                let tbl = tblBuf.baseAddress!
                var m = 0
                while m < outCount {
                    let center = Float(m) * ratio
                    let kStart = max(0, Int(ceilf(center - radius)))
                    let kEnd = min(inCount - 1, Int(floorf(center + radius)))
                    var acc: Float = 0.0
                    var wSum: Float = 0.0
                    var k = kStart
                    while k <= kEnd {
                        let idx = Int(abs(Float(k) - center) * tableScale)
                        if idx <= tableLimit {
                            let w = tbl[idx]
                            acc += src[k] * w
                            wSum += w
                        }
                        k += 1
                    }
                    if 1e-12 < wSum {
                        out[m] = acc / wSum
                    }
                    m += 1
                }
            }
        }
        return out
    }

    /// Offline feature extraction: the gain is fixed from the RMS of the first `rmsSampleCount` samples (the whole clip
    /// by default), then every hop is pushed through `StreamingFeatureFrontEnd`.
    public static func extractFeaturesFromPCM(pcmData: [Float], rmsSampleCount: Int? = nil) -> [[Float]] {
        let totalSamples = pcmData.count
        if totalSamples < 400 {
            return []
        }

        let rmsCount = min(totalSamples, max(1, rmsSampleCount ?? totalSamples))
        var sumSquares: Float = 0.0
        var rIdx = 0
        while rIdx < rmsCount {
            sumSquares += pcmData[rIdx] * pcmData[rIdx]
            rIdx += 1
        }
        let rms = sqrtf(sumSquares / Float(rmsCount))
        let front = StreamingFeatureFrontEnd()
        front.setGain(StreamingFeatureFrontEnd.gainForRMS(rms))
        front.beginUtterance()

        let frameSize = front.frameSize
        let hopSize = front.hopSize
        var out: [[Float]] = []
        var offset = 0
        pcmData.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress!
            while (offset + frameSize) <= totalSamples {
                if let stacked = front.pushRawFrame(pcmPtr: base.advanced(by: offset), count: frameSize) {
                    out.append(Array(stacked))
                }
                offset += hopSize
            }
        }
        if let last = front.flush() {
            out.append(Array(last))
        }
        return out
    }
}
