import Foundation

/// Splits long audio into utterance spans: cuts at VAD silences, merges short gaps, and splits
/// over-long spans at their quietest window. Shared by `segment` (training data) and `transcribe`.
///
/// The VAD expects RMS-normalized input; the gain is applied per frame so the input array is
/// never copied, even for 12 hours of audio.
public struct SpeechChunker: Sendable {
    /// Gaps shorter than this (seconds) are merged into one span.
    public let mergeGapSeconds: Float
    /// Spans shorter than this (seconds) are dropped.
    public let minSegmentSeconds: Float
    /// Spans longer than this (seconds) are split at the lowest-energy window.
    public let maxSegmentSeconds: Float
    public let dspConfig: DSPConfig

    public struct Span: Sendable, Equatable {
        public var start: Int
        public var end: Int
        public init(start: Int, end: Int) {
            self.start = start
            self.end = end
        }
    }

    public init(
        mergeGapSeconds: Float = 0.35,
        minSegmentSeconds: Float = 0.6,
        maxSegmentSeconds: Float = 15.0,
        dspConfig: DSPConfig = DSPConfig()
    ) {
        self.mergeGapSeconds = mergeGapSeconds
        self.minSegmentSeconds = minSegmentSeconds
        self.maxSegmentSeconds = max(2.0, maxSegmentSeconds)
        self.dspConfig = dspConfig
    }

    /// Returned spans are sample indices into `pcm`.
    public func chunk(pcm: [Float]) -> [Span] {
        let sampleRate = Float(dspConfig.sampleRate)
        let frameSize = dspConfig.frameSize
        let hopSize = dspConfig.hopSize
        if pcm.count < frameSize {
            return []
        }

        var sumSq: Float = 0.0
        var i = 0
        while i < pcm.count {
            sumSq += pcm[i] * pcm[i]
            i += 1
        }
        let gain = StreamingFeatureFrontEnd.gainForRMS(sqrt(sumSq / Float(pcm.count)))

        let vad = VAD(config: dspConfig)
        let workspace = DSPWorkspace(lpcOrder: dspConfig.lpcOrder, melChannels: dspConfig.melChannels)
        var frame = [Float](repeating: 0.0, count: frameSize)

        // Same state machine as VAD.segmentUtterances, inlined to apply the gain per frame.
        let minSpeechFrames = 5
        let hangoverFrames = 20
        let rollSamples = Int(0.15 * sampleRate)
        var merged: [Span] = []
        var state = VAD.State.silence
        var speechStart = 0

        func close(at offset: Int) {
            let end = min(pcm.count, offset + frameSize + rollSamples)
            if let last = merged.last, Float(speechStart - last.end) / sampleRate < mergeGapSeconds {
                merged[merged.count - 1].end = max(last.end, end)
            } else {
                merged.append(Span(start: speechStart, end: end))
            }
        }

        pcm.withUnsafeBufferPointer { src in
            frame.withUnsafeMutableBufferPointer { dst in
                let base = src.baseAddress!
                let buf = dst.baseAddress!
                var offset = 0
                while (offset + frameSize) <= pcm.count {
                    var k = 0
                    while k < frameSize {
                        buf[k] = base[offset + k] * gain
                        k += 1
                    }
                    let isSpeech = vad.processFrame(ptr: buf, count: frameSize, workspace: workspace).isSpeech
                    switch state {
                    case .silence:
                        if isSpeech {
                            state = .speechTriggered(triggerCount: 1)
                            speechStart = max(0, offset - rollSamples)
                        }
                    case .speechTriggered(let count):
                        if isSpeech {
                            if minSpeechFrames <= (count + 1) {
                                state = .activeSpeech(speechCount: count + 1)
                            } else {
                                state = .speechTriggered(triggerCount: count + 1)
                            }
                        } else {
                            state = .silence
                        }
                    case .activeSpeech(let count):
                        if isSpeech {
                            state = .activeSpeech(speechCount: count + 1)
                        } else {
                            state = .hangover(hangoverCount: 1)
                        }
                    case .hangover(let count):
                        if isSpeech {
                            state = .activeSpeech(speechCount: count + 1)
                        } else {
                            if hangoverFrames <= (count + 1) {
                                close(at: offset)
                                state = .silence
                            } else {
                                state = .hangover(hangoverCount: count + 1)
                            }
                        }
                    }
                    offset += hopSize
                }
                switch state {
                case .activeSpeech, .hangover:
                    close(at: pcm.count)
                default:
                    break
                }
            }
        }

        var result: [Span] = []
        var queue = merged
        while queue.isEmpty != true {
            let span = queue.removeFirst()
            let seconds = Float(span.end - span.start) / sampleRate
            if maxSegmentSeconds < seconds {
                let cut = quietestSplit(span, in: pcm, sampleRate: sampleRate)
                if span.start < cut && cut < span.end {
                    queue.insert(Span(start: cut, end: span.end), at: 0)
                    queue.insert(Span(start: span.start, end: cut), at: 0)
                    continue
                }
            }
            if minSegmentSeconds <= seconds {
                result.append(span)
            }
        }
        return result
    }

    /// Center of the lowest-energy 200 ms window, at least 1 s from either end. Gain is irrelevant for the comparison.
    private func quietestSplit(_ span: Span, in samples: [Float], sampleRate: Float) -> Int {
        let window = Int(0.2 * sampleRate)
        let hop = Int(0.05 * sampleRate)
        let margin = Int(1.0 * sampleRate)
        var best = (span.start + span.end) / 2
        var bestEnergy = Float.greatestFiniteMagnitude
        var pos = span.start + margin
        while (pos + window) <= (span.end - margin) {
            var e: Float = 0.0
            var k = pos
            while k < pos + window {
                e += samples[k] * samples[k]
                k += 1
            }
            if e < bestEnergy {
                bestEnergy = e
                best = pos + window / 2
            }
            pos += hop
        }
        return best
    }
}
