import Foundation

/// Bank of background noise (game audio, BGM, sound effects) mixed into clean utterances during training.
///
/// Clips from a JSONL manifest (`{"path": ...}`) are held as 16 kHz mono PCM. With probability `probability`,
/// one random clip is scaled to an SNR drawn from `snrRange` relative to the utterance RMS and added.
public final class NoiseBank: @unchecked Sendable {
    public static let sampleRate = 16000
    /// Probability that an utterance gets noise.
    public static let probability: Double = 0.5
    /// SNR range in dB, sampled uniformly.
    public static let snrRange: ClosedRange<Float> = 5.0...20.0

    private let clips: [[Float]]

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
    public init(clips: [[Float]]) {
        self.clips = clips.filter { $0.isEmpty != true }
    }

    /// Loads the WAV files listed in a JSONL manifest (`path` key), resampled to 16 kHz. Unreadable files are skipped.
    public convenience init(manifestPath: String) throws {
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
        self.init(clips: clips)
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
        if Self.probability < Double.random(in: 0.0..<1.0, using: &rng) {
            return speech
        }
        let snrDB = Float.random(in: Self.snrRange, using: &rng)
        let clip = clips[Int.random(in: 0..<clips.count, using: &rng)]
        let offset = Int.random(in: 0..<clip.count, using: &rng)
        return Self.mix(speech: speech, noise: clip, offset: offset, snrDB: snrDB)
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
