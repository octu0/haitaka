import Foundation

/// Training-only PCM augmentation: background noise, then read-aloud bot voices.
public final class FeatureAugmenter: @unchecked Sendable {
    public let noiseBank: NoiseBank?
    /// Read-aloud bot voices laid over the utterance after the background noise (`NoiseBank.voice`).
    public let voiceBank: NoiseBank?
    /// Random choices are seeded by the utterance path and this value, so a run is reproducible while the
    /// training loop still gives every utterance a different augmentation each epoch by changing the salt.
    public var epochSalt: UInt64 = 0
    /// Directory names of the datasets of clean recordings. When set, only rows under one of them are augmented, always
    /// as clean recordings, and every other row is left as recorded. Empty: every row, clean or not measured from the audio.
    public let cleanDatasets: Set<String>

    public init(noiseBank: NoiseBank? = nil, voiceBank: NoiseBank? = nil, cleanDatasets: Set<String> = []) {
        self.noiseBank = noiseBank
        self.voiceBank = voiceBank
        self.cleanDatasets = cleanDatasets
    }

    /// Whether a path (or augmentation key) lies under one of `cleanDatasets`.
    public func isInCleanDataset(_ path: String) -> Bool {
        for component in path.split(separator: "/") {
            if cleanDatasets.contains(String(component)) {
                return true
            }
        }
        return false
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
        if noiseBank == nil && voiceBank == nil {
            return nil
        }
        return { path, pcm in self.transformPCM(path: path, pcm) }
    }

    func transformPCM(path: String, _ pcm: [Float]) -> [Float] {
        var clean: Bool? = nil
        if cleanDatasets.isEmpty != true {
            if isInCleanDataset(path) != true {
                return pcm
            }
            clean = true
        }
        var rng = SplitMix64(seed: Self.seed(forPath: path) ^ epochSalt)
        var out = pcm
        if let bank = noiseBank {
            out = bank.mix(out, clean: clean, using: &rng)
        }
        if let bank = voiceBank {
            out = bank.mix(out, clean: clean, using: &rng)
        }
        return out
    }
}
