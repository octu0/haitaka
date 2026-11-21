import Foundation

/// Bank of interfering audio mixed into utterances during training while the label stays the utterance's own text.
///
/// Clips from a JSONL manifest (`{"path": ...}`) are held as 16 kHz mono PCM. A clean recording (its quietest frames
/// `cleanFloorDB` or more below its voiced frames: TTS, read speech) gets noise with probability `probability`, a
/// recording that already carries background sound (streams, videos) with `recordedProbability`. Then a target SNR is drawn from `snrRange` and one random clip is added so the utterance reaches it. The SNR is measured
/// on the voiced frames (pauses do not lower the reference), and the noise already in the utterance counts toward it:
/// a clean recording gets the whole amount, a stream that already carries game audio gets only what is missing, and an
/// utterance already noisier than the target is left as it is.
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
    /// Probability that a clean recording gets noise, and that a recording with its own background sound does.
    public static let probability: Double = 0.7
    public static let recordedProbability: Double = 0.25
    /// Voiced-to-floor ratio in dB from which a recording counts as clean.
    public static let cleanFloorDB: Float = 40.0
    /// Target SNR range in dB over the voiced frames, sampled uniformly.
    public static let snrRange: ClosedRange<Float> = 3.0...20.0
    /// Levels are measured on frames of this many samples (20 ms).
    public static let levelFrame = 320
    /// Frames within this ratio of the 95th-percentile frame energy (-10 dB) count as voiced.
    public static let voicedRatio: Float = 0.1
    /// The quietest fraction of frames gives the noise already in the utterance.
    public static let floorFraction: Float = 0.1
    /// Bot voices are as loud as the streamer or a little quieter.
    public static let voiceProbability: Double = 0.3
    public static let voiceSNRRange: ClosedRange<Float> = 0.0...10.0

    private let clips: [[Float]]
    public let probability: Double
    public let recordedProbability: Double
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
                recordedProbability: Double = NoiseBank.recordedProbability,
                snrRange: ClosedRange<Float> = NoiseBank.snrRange, placement: Placement = .tiled) {
        self.clips = clips.filter { $0.isEmpty != true }
        self.probability = probability
        self.recordedProbability = recordedProbability
        self.snrRange = snrRange
        self.placement = placement
    }

    /// A bank of read-aloud bot voices: placed once at `voiceSNRRange` with `voiceProbability`.
    public static func voice(clips: [[Float]]) -> NoiseBank {
        return NoiseBank(clips: clips, probability: voiceProbability, recordedProbability: voiceProbability,
                         snrRange: voiceSNRRange, placement: .once)
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

    /// `clean` set by the caller decides the probability; nil measures it from the noise floor (`cleanFloorDB`).
    public func mix<G: RandomNumberGenerator>(_ speech: [Float], clean: Bool? = nil, using rng: inout G) -> [Float] {
        if clips.isEmpty || speech.isEmpty {
            return speech
        }
        let level = Self.levels(speech)
        var isClean = level.floor * powf(10.0, Self.cleanFloorDB / 10.0) <= level.voiced
        if let given = clean {
            isClean = given
        }
        var chance = recordedProbability
        if isClean {
            chance = probability
        }
        if chance < Double.random(in: 0.0..<1.0, using: &rng) {
            return speech
        }
        let snrDB = Float.random(in: snrRange, using: &rng)
        let clip = clips[Int.random(in: 0..<clips.count, using: &rng)]
        if placement == .once {
            let start = Int.random(in: -(clip.count / 2)...max(-(clip.count / 2), speech.count - clip.count / 2), using: &rng)
            return Self.overlay(speech: speech, voice: clip, start: start, snrDB: snrDB)
        }
        let offset = Int.random(in: 0..<clip.count, using: &rng)
        let missing = level.voiced / powf(10.0, snrDB / 10.0) - level.floor
        if missing <= 0.0 {
            return speech
        }
        return Self.add(speech: speech, noise: clip, offset: offset, power: missing)
    }

    /// Mean energy of the voiced frames and of the quietest frames (the noise floor already present).
    public static func levels(_ x: [Float]) -> (voiced: Float, floor: Float) {
        var energies: [Float] = []
        var start = 0
        while start < x.count {
            let end = min(x.count, start + levelFrame)
            var acc: Float = 0.0
            var i = start
            while i < end {
                acc += x[i] * x[i]
                i += 1
            }
            energies.append(acc / Float(end - start))
            start = end
        }
        if energies.isEmpty {
            return (0.0, 0.0)
        }
        let sorted = energies.sorted()
        let loud = sorted[min(sorted.count - 1, Int(Float(sorted.count - 1) * 0.95))]
        var voicedSum: Float = 0.0
        var voicedCount = 0
        for e in energies {
            if loud * voicedRatio <= e {
                voicedSum += e
                voicedCount += 1
            }
        }
        let floorCount = max(1, Int(Float(sorted.count) * floorFraction))
        var floorSum: Float = 0.0
        var i = 0
        while i < floorCount {
            floorSum += sorted[i]
            i += 1
        }
        return (voicedSum / Float(max(1, voicedCount)), floorSum / Float(floorCount))
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

    /// Deterministic mix at an SNR against the whole-utterance RMS. `noise` is read from `offset` and wraps around.
    public static func mix(speech: [Float], noise: [Float], offset: Int, snrDB: Float) -> [Float] {
        let speechRMS = rms(speech)
        if speechRMS <= 1e-6 {
            return speech
        }
        return add(speech: speech, noise: noise, offset: offset, power: speechRMS * speechRMS / powf(10.0, snrDB / 10.0))
    }

    /// Adds `noise`, read from `offset` and wrapped around, scaled to mean energy `power` over the utterance.
    public static func add(speech: [Float], noise: [Float], offset: Int, power: Float) -> [Float] {
        if noise.isEmpty || speech.isEmpty || power <= 0.0 {
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
        let gain = power.squareRoot() / noiseRMS
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
