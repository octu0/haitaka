import Foundation

/// Training-only augmentation: speed perturbation and SpecAugment.
///
/// - Speed perturbation resamples PCM by 0.9-1.1x with linear interpolation (pitch shifts too, as in Kaldi).
/// - SpecAugment zeroes time spans (frames) and frequency bands (mel bands). Each frame stacks `stack` hops of
///   [smoothed mel 64 + delta 64] at 10 ms, so a frequency mask covers the same band in every hop and in both halves.
public final class FeatureAugmenter: @unchecked Sendable {
    public static let speedRange: ClosedRange<Float> = 0.9...1.1
    /// Number of time masks; each is at most the smaller of the frame cap and the ratio of the sequence length.
    public static let timeMasks = 2
    public static let maxTimeMaskFrames = 20
    public static let maxTimeMaskRatio: Float = 0.1
    /// Number of frequency masks and the maximum width in mel bands (out of 64).
    public static let frequencyMasks = 2
    public static let maxFrequencyMaskBands = 12

    public let speed: Bool
    public let specAugment: Bool
    public let noiseBank: NoiseBank?
    /// Random choices are seeded by the utterance path and this value, so a run is reproducible while the
    /// training loop still gives every utterance a different augmentation each epoch by changing the salt.
    public var epochSalt: UInt64 = 0

    public init(speed: Bool, specAugment: Bool, noiseBank: NoiseBank? = nil) {
        self.speed = speed
        self.specAugment = specAugment
        self.noiseBank = noiseBank
    }

    /// FNV-1a hash of the path: the per-utterance part of the augmentation seed.
    public static func seed(forPath path: String) -> UInt64 {
        var h: UInt64 = 0xCBF29CE484222325
        for byte in path.utf8 {
            h ^= UInt64(byte)
            h = h &* 0x100000001B3
        }
        return h
    }

    /// nil when no PCM augmentation is enabled, so callers can skip the hook.
    public var pcmTransform: ((String, [Float]) -> [Float])? {
        if speed != true && noiseBank == nil {
            return nil
        }
        return { path, pcm in self.transformPCM(path: path, pcm) }
    }

    public var featureTransform: ((String, [[Float]]) -> [[Float]])? {
        if specAugment != true {
            return nil
        }
        return { path, features in
            var rng = SplitMix64(seed: (Self.seed(forPath: path) ^ self.epochSalt) &+ 1)
            return Self.specAugment(features, using: &rng)
        }
    }

    func transformPCM(path: String, _ pcm: [Float]) -> [Float] {
        var rng = SplitMix64(seed: Self.seed(forPath: path) ^ epochSalt)
        var out = pcm
        if speed {
            let factor = Float.random(in: Self.speedRange, using: &rng)
            out = Self.resample(out, speed: factor)
        }
        if let bank = noiseBank {
            out = bank.mix(out, using: &rng)
        }
        return out
    }

    /// Resamples to `speed` times faster (1.1 yields 1/1.1 as many samples) with linear interpolation.
    public static func resample(_ pcm: [Float], speed: Float) -> [Float] {
        if pcm.count < 2 || speed <= 0.0 || abs(speed - 1.0) < 1e-4 {
            return pcm
        }
        let outCount = max(1, Int(Float(pcm.count) / speed))
        var out = [Float](repeating: 0.0, count: outCount)
        var i = 0
        while i < outCount {
            let position = Float(i) * speed
            let base = Int(position)
            if pcm.count - 1 <= base {
                out[i] = pcm[pcm.count - 1]
            } else {
                let frac = position - Float(base)
                out[i] = pcm[base] * (1.0 - frac) + pcm[base + 1] * frac
            }
            i += 1
        }
        return out
    }

    /// Applies SpecAugment with random mask positions and widths; masked values are 0.
    public static func specAugment<G: RandomNumberGenerator>(_ features: [[Float]], using rng: inout G) -> [[Float]] {
        let frames = features.count
        if frames == 0 {
            return features
        }
        let dim = features[0].count
        let tapDim = StreamingFeatureFrontEnd.tapDim
        let mel = StreamingFeatureFrontEnd.melChannels
        let stack = max(1, dim / tapDim)
        var out = features

        let maxTime = min(maxTimeMaskFrames, max(1, Int(Float(frames) * maxTimeMaskRatio)))
        var m = 0
        while m < timeMasks {
            let length = Int.random(in: 0...maxTime, using: &rng)
            if 0 < length && length < frames {
                let start = Int.random(in: 0...(frames - length), using: &rng)
                var t = start
                while t < start + length {
                    out[t] = [Float](repeating: 0.0, count: dim)
                    t += 1
                }
            }
            m += 1
        }

        m = 0
        while m < frequencyMasks {
            let bands = Int.random(in: 0...maxFrequencyMaskBands, using: &rng)
            if 0 < bands && bands < mel {
                let startBand = Int.random(in: 0...(mel - bands), using: &rng)
                var t = 0
                while t < frames {
                    var s = 0
                    while s < stack {
                        var b = startBand
                        while b < startBand + bands {
                            let melIndex = s * tapDim + b
                            let deltaIndex = s * tapDim + mel + b
                            if melIndex < dim {
                                out[t][melIndex] = 0.0
                            }
                            if deltaIndex < dim {
                                out[t][deltaIndex] = 0.0
                            }
                            b += 1
                        }
                        s += 1
                    }
                    t += 1
                }
            }
            m += 1
        }
        return out
    }
}
