import Foundation

/// Bank of interfering audio mixed into utterances during training while the label stays the utterance's own text.
///
/// Clips from a JSONL manifest (`{"path": ...}`) are held as 16 kHz mono PCM. With probability `probability`,
/// one random clip is scaled to an SNR drawn from `snrRange` relative to the utterance RMS and added.
/// Background noise (game audio, BGM, sound effects) is tiled over the whole utterance. Read-aloud bot voices
/// (`voice`) are placed once at a random position, as a bot reads a comment over the streamer.
public final class NoiseBank: @unchecked Sendable {
    public enum Placement: Sendable {
        /// The clip starts at a random offset and repeats to cover the utterance.
        case tiled
        /// The clip is added once at a random position; it may run past either end of the utterance.
        case once
    }

    public static let sampleRate = 16000
    /// Probability that an utterance gets noise.
    public static let probability: Double = 0.5
    /// SNR range in dB, sampled uniformly.
    public static let snrRange: ClosedRange<Float> = 5.0...20.0
    /// Bot voices are as loud as the streamer or a little quieter.
    public static let voiceProbability: Double = 0.3
    public static let voiceSNRRange: ClosedRange<Float> = 0.0...10.0

    private let clips: [[Float]]
    public let probability: Double
    public let snrRange: ClosedRange<Float>
    public let placement: Placement

    public var count: Int {
        return clips.count
    }

    public var totalSeconds: Double {
        var n = 0
        for clip in clips {
            n += clip.count
        }
        return Double(n) / Double(Self.sampleRate)
    }

    /// `clips` are 16 kHz mono PCM. Empty clips are dropped.
    public init(clips: [[Float]], probability: Double = NoiseBank.probability,
                snrRange: ClosedRange<Float> = NoiseBank.snrRange, placement: Placement = .tiled) {
        self.clips = clips.filter { $0.isEmpty != true }
        self.probability = probability
        self.snrRange = snrRange
        self.placement = placement
    }

    /// A bank of read-aloud bot voices: placed once at `voiceSNRRange` with `voiceProbability`.
    public static func voice(clips: [[Float]]) -> NoiseBank {
        return NoiseBank(clips: clips, probability: voiceProbability, snrRange: voiceSNRRange, placement: .once)
    }

    /// Loads the WAV files listed in a JSONL manifest (`path` key), resampled to 16 kHz. Unreadable files are skipped.
    public static func loadClips(manifestPath: String) throws -> [[Float]] {
        let content = try String(contentsOfFile: manifestPath, encoding: .utf8)
        var clips: [[Float]] = []
        for line in content.split(separator: "\n") {
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let path = obj["path"] as? String else {
                continue
            }
            guard let wav = SpeechDataset.loadWavFile(path: path) else {
                continue
            }
            let pcm = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
            if pcm.isEmpty != true {
                clips.append(pcm)
            }
        }
        return clips
    }

    public convenience init(manifestPath: String) throws {
        self.init(clips: try Self.loadClips(manifestPath: manifestPath))
    }

    /// Mixes noise into 16 kHz speech PCM. Returns the input unchanged when skipped by probability or when the speech is silent.
    public func mix(_ speech: [Float]) -> [Float] {
        var rng = SystemRandomNumberGenerator()
        return mix(speech, using: &rng)
    }

    public func mix<G: RandomNumberGenerator>(_ speech: [Float], using rng: inout G) -> [Float] {
        if clips.isEmpty || speech.isEmpty {
            return speech
        }
        if probability < Double.random(in: 0.0..<1.0, using: &rng) {
            return speech
        }
        let snrDB = Float.random(in: snrRange, using: &rng)
        let clip = clips[Int.random(in: 0..<clips.count, using: &rng)]
        if placement == .once {
            let start = Int.random(in: -(clip.count / 2)...max(-(clip.count / 2), speech.count - clip.count / 2), using: &rng)
            return Self.overlay(speech: speech, voice: clip, start: start, snrDB: snrDB)
        }
        let offset = Int.random(in: 0..<clip.count, using: &rng)
        return Self.mix(speech: speech, noise: clip, offset: offset, snrDB: snrDB)
    }

    /// Adds `voice` once, its first sample at `start` (negative starts cut its head, and its tail is cut at the end).
    /// The gain sets the voice's own RMS to speechRMS / 10^(SNR/20), so the SNR is the level while the voice sounds.
    public static func overlay(speech: [Float], voice: [Float], start: Int, snrDB: Float) -> [Float] {
        if voice.isEmpty || speech.isEmpty {
            return speech
        }
        let speechRMS = rms(speech)
        let voiceRMS = rms(voice)
        if speechRMS <= 1e-6 || voiceRMS <= 1e-6 {
            return speech
        }
        let gain = speechRMS / voiceRMS / powf(10.0, snrDB / 20.0)
        var out = speech
        var j = max(0, -start)
        while j < voice.count && start + j < out.count {
            out[start + j] += voice[j] * gain
            j += 1
        }
        return out
    }

    /// Deterministic mix. `noise` is read from `offset` and wraps around as needed.
    /// Noise gain is speechRMS / noiseRMS / 10^(SNR/20).
    public static func mix(speech: [Float], noise: [Float], offset: Int, snrDB: Float) -> [Float] {
        if noise.isEmpty || speech.isEmpty {
            return speech
        }
        let speechRMS = rms(speech)
        if speechRMS <= 1e-6 {
            return speech
        }
        var segment = [Float](repeating: 0.0, count: speech.count)
        var j = offset % noise.count
        var i = 0
        while i < speech.count {
            segment[i] = noise[j]
            j += 1
            if noise.count <= j {
                j = 0
            }
            i += 1
        }
        let noiseRMS = rms(segment)
        if noiseRMS <= 1e-6 {
            return speech
        }
        let gain = speechRMS / noiseRMS / powf(10.0, snrDB / 20.0)
        var out = speech
        i = 0
        while i < out.count {
            out[i] += segment[i] * gain
            i += 1
        }
        return out
    }

    static func rms(_ x: [Float]) -> Float {
        if x.isEmpty {
            return 0.0
        }
        var acc: Double = 0.0
        for v in x {
            acc += Double(v) * Double(v)
        }
        return Float((acc / Double(x.count)).squareRoot())
    }
}
