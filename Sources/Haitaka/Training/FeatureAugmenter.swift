import Foundation

/// Training-only PCM augmentation: speed perturbation, then background noise and read-aloud bot voices.
///
/// Speed perturbation resamples PCM by 0.9-1.1x with linear interpolation (pitch shifts too, as in Kaldi).
public final class FeatureAugmenter: @unchecked Sendable {
    public static let speedRange: ClosedRange<Float> = 0.9...1.1

    public let speed: Bool
    public let noiseBank: NoiseBank?
    /// Read-aloud bot voices laid over the utterance after the background noise (`NoiseBank.voice`).
    public let voiceBank: NoiseBank?
    /// Random choices are seeded by the utterance path and this value, so a run is reproducible while the
    /// training loop still gives every utterance a different augmentation each epoch by changing the salt.
    public var epochSalt: UInt64 = 0

    public init(speed: Bool, noiseBank: NoiseBank? = nil, voiceBank: NoiseBank? = nil) {
        self.speed = speed
        self.noiseBank = noiseBank
        self.voiceBank = voiceBank
    }

    /// Augmentation key of a manifest row. A manifest may list the same path more than once (data repeated to weight
    /// it), and each row should hear different noise, so the row index is part of the key.
    public static func key(path: String, index: Int) -> String {
        return "\(path)#\(index)"
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
        if speed != true && noiseBank == nil && voiceBank == nil {
            return nil
        }
        return { path, pcm in self.transformPCM(path: path, pcm) }
    }

    /// The speed factor this epoch's PCM transform applies to the utterance (1 without speed perturbation). It is the
    /// transform's first random draw, so batches can be formed on the perturbed lengths before the audio is read.
    public func speedFactor(path: String) -> Float {
        if speed != true {
            return 1.0
        }
        var rng = SplitMix64(seed: Self.seed(forPath: path) ^ epochSalt)
        return Float.random(in: Self.speedRange, using: &rng)
    }

    /// Feature frames of an utterance of `frames` frames after this epoch's speed perturbation.
    public func frameCount(_ frames: Int, path: String) -> Int {
        let factor = speedFactor(path: path)
        if abs(factor - 1.0) < 1e-4 {
            return frames
        }
        return max(1, Int(Float(frames) / factor))
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
        if let bank = voiceBank {
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
}
