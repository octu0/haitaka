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

    /// `pcmTransform` runs on 16 kHz PCM before feature extraction (speed perturbation, noise mixing) and
    /// `featureTransform` on the extracted features (SpecAugment). Both are training-only.
    public static func loadFeatures(
        path: String,
        pcmTransform: ((String, [Float]) -> [Float])? = nil,
        featureTransform: ((String, [[Float]]) -> [[Float]])? = nil
    ) -> (pcm: [Float], features: [[Float]]) {
        guard let wav = loadWavFile(path: path) else {
            return ([], [])
        }
        var pcm16k = resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
        if let transform = pcmTransform {
            pcm16k = transform(path, pcm16k)
        }
        var features = extractFeaturesFromPCM(pcmData: pcm16k)
        if let transform = featureTransform {
            features = transform(path, features)
        }
        return (pcm16k, features)
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

    /// Resamples to 16 kHz. 48 kHz uses a 3-tap box filter before decimation; other rates use linear interpolation.
    public static func resampleTo16k(pcmData: [Float], sampleRate: Int) -> [Float] {
        if sampleRate == 16000 || pcmData.isEmpty {
            return pcmData
        }

        if sampleRate == 48000 {
            let outCount = pcmData.count / 3
            var out = [Float](repeating: 0.0, count: outCount)
            var m = 0
            while m < outCount {
                let srcIdx = m * 3
                let s0 = pcmData[srcIdx]
                var s1 = s0
                if (srcIdx + 1) < pcmData.count {
                    s1 = pcmData[srcIdx + 1]
                }
                var s2 = s1
                if (srcIdx + 2) < pcmData.count {
                    s2 = pcmData[srcIdx + 2]
                }
                out[m] = (s0 + s1 + s2) / 3.0
                m += 1
            }
            return out
        }

        let ratio = Float(sampleRate) / 16000.0
        let outCount = Int(Float(pcmData.count) / ratio)
        if outCount <= 0 {
            return []
        }
        var out = [Float](repeating: 0.0, count: outCount)
        var m = 0
        while m < outCount {
            let srcPos = Float(m) * ratio
            let i0 = Int(srcPos)
            let i1 = min(pcmData.count - 1, i0 + 1)
            let frac = srcPos - Float(i0)
            if i0 < pcmData.count {
                out[m] = (1.0 - frac) * pcmData[i0] + frac * pcmData[i1]
            }
            m += 1
        }
        return out
    }

    /// Offline feature extraction: the gain is fixed from the clip RMS, then every hop is pushed through
    /// `StreamingFeatureFrontEnd`.
    public static func extractFeaturesFromPCM(pcmData: [Float]) -> [[Float]] {
        let totalSamples = pcmData.count
        if totalSamples < 400 {
            return []
        }

        var sumSquares: Float = 0.0
        var rIdx = 0
        // Experiment switch (10-19): HAITAKA_DSP_RANGE=1 measures the RMS after pre-emphasis, so the gain levels the
        // signal the filterbank actually sees instead of the raw waveform dominated by the low band.
        let preemphasizedGain = ProcessInfo.processInfo.environment["HAITAKA_DSP_PREGAIN"] == "1"
        while rIdx < totalSamples {
            var x = pcmData[rIdx]
            if preemphasizedGain && 0 < rIdx {
                x = pcmData[rIdx] - 0.97 * pcmData[rIdx - 1]
            }
            sumSquares += x * x
            rIdx += 1
        }
        let rms = sqrtf(sumSquares / Float(totalSamples))
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
