@preconcurrency import AVFoundation
import CoreAudio
import Foundation
import Haitaka

setbuf(stdout, nil)

// Live transcription of microphone input.
//
// Audio is split into utterances at silences. Recognition has two stages: the acoustic model
// emits kana, then a dictionary Viterbi search converts to kanji. Runs HawkNetwork + CTC beam search on CPU.

// MARK: - Arguments

var weightsPath = ""
var corpusPath = ""
var kanaLMPath = ""
var silenceSeconds = 0.6      // Silence length that ends an utterance
var minSpeechSeconds = 0.3    // Shorter voiced runs are discarded as noise
var maxSpeechSeconds = 15.0   // Forced cut for continuous speech
var showKana = false
var listMicrophones = false
var showMeter = false
var micSelector = ""          // Device index or part of its name
var wavInputPath = ""         // Stream a WAV file instead of the microphone (for testing)

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-d", "--corpus":
        if (argIdx + 1) < args.count {
            corpusPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--kana-lm":
        if (argIdx + 1) < args.count {
            kanaLMPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--silence":
        if (argIdx + 1) < args.count {
            if let v = Double(args[argIdx + 1]) {
                silenceSeconds = max(0.1, v)
            }
            argIdx += 1
        }
    case "--kana":
        showKana = true
    case "--meter":
        showMeter = true
    case "--mic-list":
        listMicrophones = true
    case "--mic":
        if (argIdx + 1) < args.count {
            micSelector = args[argIdx + 1]
            argIdx += 1
        }
    case "--wav":
        if (argIdx + 1) < args.count {
            wavInputPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-h", "--help":
        print("""
        Live microphone transcription

          mictrans -w <weights.json> [-d <dictionary text>] [options]

        Options:
          --mic-list         List input devices and exit
          --mic <index|name> Input device to use (default: the system default device)
          --silence <sec>    Silence length that ends an utterance (default 0.6)
          --kana-lm <path>   Kana text, one sentence per line (kanalm output); adds an n-gram LM to the CTC beam search
          --kana             Also show the raw kana (first-stage output)
          --meter            Show the input level and VAD decision
          --wav <audio.wav>  Stream a WAV file in 10 ms blocks instead of the microphone (for testing)

        The kana vocabulary is read from the weights file.
        Pass a list of sentences with -d to enable second-stage kana-to-kanji conversion (kana only otherwise).
        """)
        exit(0)
    default:
        break
    }
    argIdx += 1
}

// MARK: - Input device enumeration

/// A CoreAudio input device.
struct AudioInputDevice {
    let id: AudioDeviceID
    let name: String
    let channels: Int
    let sampleRate: Double
}

/// Reads a scalar property. Reference types are not handled (CFString goes through getDeviceName).
func deviceProperty<T: Numeric>(
    _ deviceId: AudioDeviceID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
    initial: T
) -> T? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
    var size = UInt32(MemoryLayout<T>.size)
    var value = initial
    let status = withUnsafeMutablePointer(to: &value) { ptr in
        AudioObjectGetPropertyData(deviceId, &address, 0, nil, &size, ptr)
    }
    if status != noErr {
        return nil
    }
    return value
}

/// Reads the device name. The CFString is received through Unmanaged.
func getDeviceName(_ deviceId: AudioDeviceID) -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var unmanaged: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = withUnsafeMutablePointer(to: &unmanaged) { ptr in
        AudioObjectGetPropertyData(deviceId, &address, 0, nil, &size, ptr)
    }
    if status != noErr {
        return "(unknown)"
    }
    switch unmanaged {
    case .some(let ref):
        return ref.takeRetainedValue() as String
    case .none:
        return "(unknown)"
    }
}

/// Returns only devices that have input channels.
func listInputDevices() -> [AudioInputDevice] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var dataSize: UInt32 = 0
    if AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize) != noErr {
        return []
    }
    let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var ids = [AudioDeviceID](repeating: 0, count: count)
    if AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize, &ids) != noErr {
        return []
    }

    var results: [AudioInputDevice] = []
    for deviceId in ids {
        var streamAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var streamSize: UInt32 = 0
        if AudioObjectGetPropertyDataSize(deviceId, &streamAddress, 0, nil, &streamSize) != noErr {
            continue
        }
        if streamSize == 0 {
            continue
        }
        let bufferList = UnsafeMutableRawPointer.allocate(
            byteCount: Int(streamSize), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { bufferList.deallocate() }
        if AudioObjectGetPropertyData(
            deviceId, &streamAddress, 0, nil, &streamSize, bufferList) != noErr {
            continue
        }
        let lists = UnsafeMutableAudioBufferListPointer(
            bufferList.assumingMemoryBound(to: AudioBufferList.self))
        var channels = 0
        for buffer in lists {
            channels += Int(buffer.mNumberChannels)
        }
        if channels == 0 {
            continue
        }

        let name = getDeviceName(deviceId)
        let rate = deviceProperty(
            deviceId,
            selector: kAudioDevicePropertyNominalSampleRate,
            scope: kAudioObjectPropertyScopeGlobal,
            initial: Float64(0)
        ) ?? 0
        results.append(AudioInputDevice(
            id: deviceId, name: name, channels: channels, sampleRate: rate))
    }
    return results
}

func defaultInputDeviceId() -> AudioDeviceID? {
    return deviceProperty(
        AudioObjectID(kAudioObjectSystemObject),
        selector: kAudioHardwarePropertyDefaultInputDevice,
        scope: kAudioObjectPropertyScopeGlobal,
        initial: AudioDeviceID(0)
    )
}

if listMicrophones {
    let devices = listInputDevices()
    if devices.isEmpty {
        print("No input devices found.")
        exit(1)
    }
    let defaultId = defaultInputDeviceId()
    print("Input devices:")
    var i = 0
    while i < devices.count {
        let d = devices[i]
        let mark = (d.id == defaultId) ? " [default]" : ""
        print(String(format: "  %d: %@ (%d ch, %.0f Hz)%@",
                     i, d.name, d.channels, d.sampleRate, mark))
        i += 1
    }
    print("")
    print("Select one with --mic <index> or --mic <part of the name>.")
    exit(0)
}

if weightsPath.isEmpty {
    print("Usage: mictrans -w <weights.json> [-d <dictionary text>] [--kana-lm <kana text>] [--mic <index>]  (see --help)")
    exit(1)
}

// MARK: - Shutdown

/// Ctrl+C handler.
///
/// Runs on a global queue: on the main queue it could not fire while the main thread is blocked
/// (building the dictionary, waiting for microphone permission), and SIG_IGN would just swallow the signal.
/// State lives in this class because top-level variables are @MainActor-isolated and
/// cannot be touched from another thread.
final class ShutdownController: @unchecked Sendable {
    private let lock = NSLock()
    private var session: MicSession? = nil
    private var seen = false

    func attach(_ session: MicSession) {
        lock.lock()
        self.session = session
        lock.unlock()
    }

    func handle() {
        lock.lock()
        if seen {
            lock.unlock()
            print("")
            print("Force quitting.")
            exit(130)
        }
        seen = true
        let current = session
        lock.unlock()

        print("")
        print("Exiting...")
        current?.finish()
        exit(0)
    }
}

/// Installs the signal source.
/// The controller is passed in so the closure does not capture top-level variables.
func installInterruptHandler(_ controller: ShutdownController) -> DispatchSourceSignal {
    let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    source.setEventHandler {
        controller.handle()
    }
    source.resume()
    signal(SIGINT, SIG_IGN)
    return source
}

let shutdown = ShutdownController()
let interruptSource = installInterruptHandler(shutdown)

// MARK: - Model and dictionary

guard let weights = try? HawkWeights.load(from: URL(fileURLWithPath: weightsPath)), let kanaVocabulary = weights.vocabulary else {
    print("Error: cannot load weights (they must embed the kana vocabulary): \(weightsPath)")
    exit(1)
}

var kanaLanguageModel: KanaNgramLM? = nil
if kanaLMPath.isEmpty != true {
    guard let lm = KanaNgramLM(contentsOfFile: kanaLMPath, vocabulary: kanaVocabulary) else {
        print("Error: cannot read kana text: \(kanaLMPath)")
        exit(1)
    }
    kanaLanguageModel = lm
}

let kanaKanjiDict = KanaKanjiDictionary()
if corpusPath.isEmpty != true {
    guard let corpusContent = try? String(contentsOfFile: corpusPath, encoding: .utf8) else {
        print("Error: cannot read text: \(corpusPath)")
        exit(1)
    }
    kanaKanjiDict.buildFromCorpus(rawTexts: corpusContent.components(separatedBy: "\n")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { $0.isEmpty != true })
}

let network = HawkNetwork(weights: weights)

print("==================================================")
print("=== Live microphone transcription ===")
print("==================================================")
print("Acoustic model: Hawk, \(weights.numLayers) layers, width \(weights.modelDim), lookahead \(weights.lookaheadFrames * 40) ms, kana vocabulary \(kanaVocabulary.size) chars")
if let lm = kanaLanguageModel {
    print("Kana \(lm.order)-gram LM: \(kanaLMPath)")
}
if 0 < kanaKanjiDict.count {
    print("Kana-kanji dictionary: \(kanaKanjiDict.count) entries")
} else {
    print("Kana-kanji dictionary: none (showing kana only)")
}

// MARK: - Recognizer (runs on its own serial queue, separate from the audio capture thread)

/// Shows the level meter and recognition result on two fixed lines.
///
/// On a terminal, rewrites them in place with ANSI cursor movement;
/// when piped, prints one line at a time.
final class StatusDisplay: @unchecked Sendable {
    private let lock = NSLock()
    private let live: Bool
    private let hasLevel: Bool
    private var drawn = false
    private var levelLine = ""
    private var resultLine = "(waiting for speech)"

    init(live: Bool, hasLevel: Bool) {
        self.live = live
        self.hasLevel = hasLevel
    }

    func setLevel(_ text: String) {
        lock.lock()
        levelLine = text
        if live {
            render()
        } else {
            print(text)
        }
        lock.unlock()
    }

    func setResult(_ text: String) {
        lock.lock()
        resultLine = text
        if live {
            render()
        } else {
            print(text)
        }
        lock.unlock()
    }

    /// Ends the fixed display so later output does not overlap it.
    func finish() {
        lock.lock()
        if live && drawn {
            print("")
        }
        drawn = false
        lock.unlock()
    }

    /// Rewrites the fixed lines (live terminal only).
    private func render() {
        var out = ""
        if drawn && hasLevel {
            // Move to the start of the first line.
            out += "\r\u{1B}[1A"
        } else {
            if drawn {
                out += "\r"
            }
        }
        if hasLevel {
            out += levelLine + "\u{1B}[K\n"
        }
        out += resultLine + "\u{1B}[K"
        drawn = true
        FileHandle.standardOutput.write(Data(out.utf8))
    }
}

/// Buffers 16 kHz PCM from the microphone, splits it at silences, and recognizes each utterance.
final class Transcriber: @unchecked Sendable {
    /// VAD window (64 ms). Much longer than the hop so autocorrelation has enough samples.
    static let vadWindowSamples = 1024
    /// Target level for input gain; matches the normalization used in training.
    static let vadTargetRMS: Float = 0.05

    private let network: HawkNetwork
    private let dictionary: KanaKanjiDictionary
    private let showKana: Bool
    private let showMeter: Bool
    private let display: StatusDisplay?

    // Streaming recognition state (accessed only on recognizeQueue).
    // Features, acoustic model, and CTC beam advance every 10 ms hop, so partial results just read
    // the current hypotheses and cost is linear in utterance length.
    private let frontEnd: StreamingFeatureFrontEnd
    private var state: HawkNetwork.State
    private let ctcDecoder: CTCDecoder
    private let kanaDecoder: KanaKanjiDecoder
    private var hopBuffer: [Float]
    private var hopFill = 0
    private var frameBuffer: [Float]
    private var haveFirstHop = false
    private var utteranceStartedAt: CFAbsoluteTime = 0.0
    private var utteranceSamples = 0

    /// Lightweight queue for buffering audio and silence detection only.
    private let audioQueue = DispatchQueue(label: "mictrans.audio")
    /// Serial queue for recognition (feature extraction, acoustic model, beam search).
    private let recognizeQueue = DispatchQueue(label: "mictrans.recognize")
    private var silenceRun = 0
    private var speechRun = 0
    private var utteranceIndex = 0
    private var meterBlocks = 0
    private var meterPeak: Float = 0.0
    private var samplesSincePartial = 0
    /// Interval between partial results (200 ms at 16 kHz).
    private let partialIntervalSamples = 3200

    private let silenceSamples: Int
    private let minSpeechSamples: Int
    private let maxSpeechSamples: Int
    /// Hop for level computation and VAD (10 ms).
    private let blockSamples = 160

    /// Multi-feature VAD (energy, zero-crossing rate, autocorrelation voicing).
    /// It tracks the background noise floor instead of using a fixed level threshold,
    /// so it picks up speech even at low microphone levels.
    private let vad: VAD
    private let dspWorkspace: DSPWorkspace
    /// Window passed to VAD; longer than the hop for autocorrelation pitch analysis.
    private var vadWindow: [Float]
    private var vadScaled: [Float]
    /// Recent peak RMS (slow decay), used to compute input gain.
    private var recentPeakRMS: Float = 0.0
    private var lastVoicingRatio: Float = 0.0
    private var lastNoiseFloor: Float = 0.0
    private var lastGain: Float = 1.0

    init(
        network: HawkNetwork,
        vocabulary: TextVocabulary,
        languageModel: KanaNgramLM?,
        dictionary: KanaKanjiDictionary,
        silenceSeconds: Double,
        minSpeechSeconds: Double,
        maxSpeechSeconds: Double,
        showKana: Bool,
        showMeter: Bool,
        display: StatusDisplay?
    ) {
        self.network = network
        self.dictionary = dictionary
        self.showKana = showKana
        self.showMeter = showMeter
        self.display = display
        self.vad = VAD(config: DSPConfig())
        self.dspWorkspace = DSPWorkspace()
        self.vadWindow = [Float](repeating: 0.0, count: Transcriber.vadWindowSamples)
        self.vadScaled = [Float](repeating: 0.0, count: Transcriber.vadWindowSamples)
        self.silenceSamples = Int(silenceSeconds * 16000.0)
        self.minSpeechSamples = Int(minSpeechSeconds * 16000.0)
        self.maxSpeechSamples = Int(maxSpeechSeconds * 16000.0)

        self.frontEnd = StreamingFeatureFrontEnd()
        self.state = network.makeState()
        self.ctcDecoder = CTCDecoder(vocabulary: vocabulary, languageModel: languageModel)
        self.kanaDecoder = KanaKanjiDecoder(dictionary: dictionary)
        self.hopBuffer = [Float](repeating: 0.0, count: frontEnd.hopSize)
        self.frameBuffer = [Float](repeating: 0.0, count: frontEnd.frameSize)
    }

    /// Called from the microphone thread. Buffering and silence detection run on audioQueue.
    func append(_ samples: [Float]) {
        audioQueue.async {
            self.consume(samples)
        }
    }

    /// Finalizes any in-progress utterance when capture ends.
    /// Waiting is bounded so a slow recognition does not block shutdown.
    func flushRemaining(timeout: TimeInterval = 2.0) {
        let gate = DispatchSemaphore(value: 0)
        audioQueue.async {
            let longEnough = (self.minSpeechSamples <= self.speechRun)
            self.speechRun = 0
            self.silenceRun = 0
            self.samplesSincePartial = 0
            self.recognizeQueue.async {
                if longEnough {
                    self.finishUtterance()
                }
                gate.signal()
            }
        }
        _ = gate.wait(timeout: .now() + timeout)
    }

    private func consume(_ samples: [Float]) {
        var offset = 0
        while offset < samples.count {
            let end = min(offset + blockSamples, samples.count)
            let block = Array(samples[offset..<end])
            offset = end

            var sumSquares: Float = 0.0
            for v in block {
                sumSquares += v * v
            }
            let rms = sqrtf(sumSquares / Float(max(1, block.count)))
            let isSpeech = judgeSpeech(block: block, rms: rms)

            if showMeter {
                if meterPeak < rms {
                    meterPeak = rms
                }
                meterBlocks += 1
                // Refresh every 10 blocks (about 0.1 s).
                if 10 <= meterBlocks {
                    let bars = Int(min(24.0, meterPeak * 400.0))
                    let bar = String(repeating: "=", count: max(0, bars))
                    var mark = "silent"
                    if 0 < speechRun {
                        mark = "voiced"
                    }
                    // The line end is cleared with an escape sequence, so no padding is needed.
                    let text = String(
                        format: "[level] %.4f x%.1f voicing %.2f floor %.5f %@ ",
                        meterPeak, lastGain, lastVoicingRatio, lastNoiseFloor, mark) + bar
                    switch display {
                    case .some(let d):
                        d.setLevel(text)
                    case .none:
                        print("  " + text)
                    }
                    meterBlocks = 0
                    meterPeak = 0.0
                }
            }

            // Leading silence before speech is not fed to the recognizer.
            if isSpeech != true && speechRun == 0 {
                continue
            }
            if speechRun == 0 {
                recognizeQueue.async {
                    self.beginUtterance()
                }
            }

            recognizeQueue.async {
                self.pushSamples(block)
            }
            if isSpeech {
                speechRun += block.count
                silenceRun = 0
            } else {
                silenceRun += block.count
            }

            let longEnough = (minSpeechSamples <= speechRun)
            let endedBySilence = (silenceSamples <= silenceRun)
            let endedByLength = (maxSpeechSamples <= (speechRun + silenceRun))
            if (longEnough && endedBySilence) || endedByLength {
                recognizeQueue.async {
                    self.finishUtterance()
                }
                speechRun = 0
                silenceRun = 0
                samplesSincePartial = 0
                continue
            }
            // Silence after only a short voiced run: discard as noise.
            if longEnough != true && endedBySilence {
                speechRun = 0
                silenceRun = 0
                samplesSincePartial = 0
                continue
            }

            // Emit partial results periodically while speaking; this only reads the current hypotheses.
            samplesSincePartial += block.count
            if longEnough && partialIntervalSamples <= samplesSincePartial {
                samplesSincePartial = 0
                recognizeQueue.async {
                    self.showPartial()
                }
            }
        }
    }

    /// Voice activity decision with adaptive VAD.
    ///
    /// To work with quiet microphones, an input gain is estimated from recent levels and applied
    /// only to the VAD window (the recognizer renormalizes with a running RMS from utterance start).
    private func judgeSpeech(block: [Float], rms: Float) -> Bool {
        // Shift the window by one hop and append the new block.
        let window = Transcriber.vadWindowSamples
        let shift = min(block.count, window)
        if shift < window {
            var i = 0
            while i < (window - shift) {
                vadWindow[i] = vadWindow[i + shift]
                i += 1
            }
        }
        var j = 0
        while j < shift {
            vadWindow[window - shift + j] = block[block.count - shift + j]
            j += 1
        }

        // Gain that moves the slowly decaying peak level toward the target level.
        let decayed = recentPeakRMS * 0.995
        if decayed < rms {
            recentPeakRMS = rms
        } else {
            recentPeakRMS = decayed
        }
        var gain: Float = 1.0
        if 1e-5 < recentPeakRMS {
            gain = min(20.0, max(1.0, Transcriber.vadTargetRMS / recentPeakRMS))
        }
        lastGain = gain

        var k = 0
        while k < window {
            vadScaled[k] = vadWindow[k] * gain
            k += 1
        }

        var result: VADResult? = nil
        vadScaled.withUnsafeBufferPointer { ptr in
            result = vad.processFrame(ptr: ptr.baseAddress!, count: window, workspace: dspWorkspace)
        }
        switch result {
        case .some(let r):
            lastVoicingRatio = r.voicingRatio
            lastNoiseFloor = r.noiseFloor
            return r.isSpeech
        case .none:
            return false
        }
    }

    // MARK: Streaming recognition (recognizeQueue)

    /// Start of an utterance. Resets features, acoustic model state, and the beam.
    private func beginUtterance() {
        frontEnd.beginUtterance()
        state = network.makeState()
        ctcDecoder.reset()
        hopFill = 0
        haveFirstHop = false
        utteranceSamples = 0
        utteranceStartedAt = CFAbsoluteTimeGetCurrent()
    }

    /// Groups audio into 10 ms hops and advances one frame per 20 ms window (previous hop + current hop).
    private func pushSamples(_ samples: [Float]) {
        utteranceSamples += samples.count
        let hop = frontEnd.hopSize
        var i = 0
        while i < samples.count {
            hopBuffer[hopFill] = samples[i]
            hopFill += 1
            i += 1
            if hopFill < hop {
                continue
            }
            hopFill = 0
            // frameBuffer = [previous hop | current hop]
            var k = 0
            while k < hop {
                frameBuffer[k] = frameBuffer[hop + k]
                frameBuffer[hop + k] = hopBuffer[k]
                k += 1
            }
            if haveFirstHop != true {
                haveFirstHop = true
                continue
            }
            let stacked = frameBuffer.withUnsafeBufferPointer { buf in
                frontEnd.pushRawFrame(pcmPtr: buf.baseAddress!, count: frontEnd.frameSize)
            }
            switch stacked {
            case .some(let features):
                advanceFrame(features)
            case .none:
                break
            }
        }
    }

    /// Feeds one stacked feature frame (about 40 ms) to the acoustic model and its log probabilities to the CTC beam.
    private func advanceFrame(_ features: [Float]) {
        ctcDecoder.push(frames: network.push([features], state: state))
    }

    /// End of an utterance. Finalizes the 3-tap filter and flushes frames held back for lookahead.
    private func flushFrontEnd() {
        if let features = frontEnd.flush() {
            advanceFrame(features)
        }
        ctcDecoder.push(frames: network.finish(state: state))
    }

    private func showPartial() {
        let kana = ctcDecoder.best.text
        if kana.isEmpty {
            return
        }
        let seconds = Double(utteranceSamples) / 16000.0
        switch display {
        case .some(let d):
            d.setResult(String(format: "[…] (%.1fs) %@", seconds, kana))
        case .none:
            break
        }
    }

    private func finishUtterance() {
        flushFrontEnd()
        let kana = ctcDecoder.best.text
        let seconds = Double(utteranceSamples) / 16000.0
        let elapsed = (CFAbsoluteTimeGetCurrent() - utteranceStartedAt) * 1000.0
        beginUtterance()
        if kana.isEmpty {
            return
        }

        let hasDictionary = (0 < dictionary.count)
        var kanji = ""
        if hasDictionary {
            kanji = kanaDecoder.decode(kanaText: kana)
        }

        utteranceIndex += 1
        let body: String
        switch (hasDictionary, showKana) {
        case (false, _):
            body = kana
        case (true, true):
            body = "\(kana) → \(kanji)"
        case (true, false):
            body = kanji
        }

        switch display {
        case .some(let d):
            d.setResult(String(format: "[%d] (%.1fs) %@", utteranceIndex, seconds, body))
        case .none:
            print(String(format: "[%d] (%.1f s, %.0f ms after speech onset)", utteranceIndex, seconds, elapsed))
            print("  → \(body)")
        }
    }
}

// On a terminal, use the in-place fixed-line display.
// --meter adds a level line.
let isTerminal = (isatty(1) == 1)
var statusDisplay: StatusDisplay? = nil
if isTerminal && wavInputPath.isEmpty {
    statusDisplay = StatusDisplay(live: true, hasLevel: showMeter)
}

let transcriber = Transcriber(
    network: network,
    vocabulary: kanaVocabulary,
    languageModel: kanaLanguageModel,
    dictionary: kanaKanjiDict,
    silenceSeconds: silenceSeconds,
    minSpeechSeconds: minSpeechSeconds,
    maxSpeechSeconds: maxSpeechSeconds,
    showKana: showKana,
    showMeter: showMeter,
    display: statusDisplay
)

// MARK: - WAV input (for testing)

if wavInputPath.isEmpty != true {
    guard let wav = SpeechDataset.loadWavFile(path: wavInputPath) else {
        print("Error: cannot read audio: \(wavInputPath)")
        exit(1)
    }
    let pcm = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
    print("WAV input: streaming \(String(format: "%.1f", Double(pcm.count) / 16000.0)) s in 10 ms blocks")
    let started = CFAbsoluteTimeGetCurrent()
    var offset = 0
    while offset < pcm.count {
        let end = min(offset + 160, pcm.count)
        transcriber.append(Array(pcm[offset..<end]))
        offset = end
    }
    transcriber.flushRemaining(timeout: 30.0)
    let elapsed = CFAbsoluteTimeGetCurrent() - started
    print(String(format: "Processing time %.2f s (RTF %.4f)", elapsed, elapsed / (Double(pcm.count) / 16000.0)))
    exit(0)
}

// MARK: - Microphone input

// Request microphone permission first. The first time, the system shows a dialog; without waiting
// here the program appears to hang silently during engine setup.
func requestMicrophoneAccess() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return true
    case .notDetermined:
        print("")
        print("Requesting microphone access. Choose \"Allow\" in the dialog...")
        let gate = DispatchSemaphore(value: 0)
        let granted = GrantBox()
        AVCaptureDevice.requestAccess(for: .audio) { ok in
            granted.value = ok
            gate.signal()
        }
        if gate.wait(timeout: .now() + 120.0) == .timedOut {
            print("No response to the microphone access request.")
            return false
        }
        return granted.value
    default:
        return false
    }
}

final class GrantBox: @unchecked Sendable {
    var value = false
}

if requestMicrophoneAccess() != true {
    print("Error: microphone access is not granted.")
    print("  Allow the terminal (or this app) in System Settings > Privacy & Security > Microphone.")
    exit(1)
}

let engine = AVAudioEngine()
let inputNode = engine.inputNode

// Switch to the requested input device before reading the format;
// the format depends on the device.
if micSelector.isEmpty != true {
    let devices = listInputDevices()
    var chosen: AudioInputDevice? = nil
    switch Int(micSelector) {
    case .some(let index):
        if 0 <= index && index < devices.count {
            chosen = devices[index]
        }
    case .none:
        let needle = micSelector.lowercased()
        for d in devices {
            if d.name.lowercased().contains(needle) {
                chosen = d
                break
            }
        }
    }
    guard let device = chosen else {
        print("Error: input device \"\(micSelector)\" not found. Run --mic-list to see the devices.")
        exit(1)
    }
    guard let audioUnit = inputNode.audioUnit else {
        print("Error: cannot switch the input device.")
        exit(1)
    }
    var deviceId = device.id
    let status = AudioUnitSetProperty(
        audioUnit,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &deviceId,
        UInt32(MemoryLayout<AudioDeviceID>.size)
    )
    if status != noErr {
        print("Error: failed to switch the input device (status \(status))")
        exit(1)
    }
    print("")
    print("Input device: \(device.name)")
}

// The tap receives the node's output format. Passing the hardware input format causes a
// channel count mismatch and AVAudioEngine crashes on an internal assertion (Trace/BPT trap).
let tapFormat = inputNode.outputFormat(forBus: 0)
if tapFormat.sampleRate <= 0 || tapFormat.channelCount == 0 {
    print("Error: the microphone is unavailable.")
    print("  Allow the terminal in System Settings > Privacy & Security > Microphone.")
    exit(1)
}

// The recognizer expects 16 kHz mono Float. Channels are downmixed here and
// AVAudioConverter is used only for sample rate conversion (mono to mono);
// how it maps multiple channels varies by environment.
guard let monoSourceFormat = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: tapFormat.sampleRate,
    channels: 1,
    interleaved: false
) else {
    print("Error: cannot create the intermediate format.")
    exit(1)
}
guard let targetFormat = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: 16000.0,
    channels: 1,
    interleaved: false
) else {
    print("Error: cannot create the output format.")
    exit(1)
}
guard let converter = AVAudioConverter(from: monoSourceFormat, to: targetFormat) else {
    print("Error: cannot convert the microphone input to 16 kHz.")
    exit(1)
}

print("")
let hwFormat = inputNode.inputFormat(forBus: 0)
print("Microphone: \(Int(tapFormat.sampleRate)) Hz \(tapFormat.channelCount) ch -> 16000 Hz 1 ch")
print("  (hardware input: \(Int(hwFormat.sampleRate)) Hz \(hwFormat.channelCount) ch)")
print("Utterances end after \(String(format: "%.1f", silenceSeconds)) s of silence (adaptive VAD: energy, zero-crossing rate, autocorrelation voicing)")
print("Start speaking. Press Ctrl+C to quit.")

/// Converts microphone input to a 16 kHz mono Float array.
/// AVAudioConverter and the input buffer are not Sendable, so conversion stays on
/// the tap thread and only the array is passed out.
final class MicConverter: @unchecked Sendable {
    private let converter: AVAudioConverter
    private let monoSourceFormat: AVAudioFormat
    private let targetFormat: AVAudioFormat
    private var reportedError = false

    init(converter: AVAudioConverter, monoSourceFormat: AVAudioFormat, targetFormat: AVAudioFormat) {
        self.converter = converter
        self.monoSourceFormat = monoSourceFormat
        self.targetFormat = targetFormat
    }

    /// Averages multichannel input to mono, then converts to 16 kHz.
    func convert(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frames = Int(buffer.frameLength)
        if frames == 0 {
            return []
        }
        guard let channels = buffer.floatChannelData else {
            return []
        }
        let channelCount = Int(buffer.format.channelCount)
        if channelCount == 0 {
            return []
        }

        guard let monoBuffer = AVAudioPCMBuffer(
            pcmFormat: monoSourceFormat,
            frameCapacity: AVAudioFrameCount(frames)
        ) else {
            return []
        }
        monoBuffer.frameLength = AVAudioFrameCount(frames)
        guard let monoChannel = monoBuffer.floatChannelData else {
            return []
        }
        let invChannels = 1.0 / Float(channelCount)
        var i = 0
        while i < frames {
            var sum: Float = 0.0
            var ch = 0
            while ch < channelCount {
                sum += channels[ch][i]
                ch += 1
            }
            monoChannel[0][i] = sum * invChannels
            i += 1
        }

        let ratio = targetFormat.sampleRate / monoSourceFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(frames) * ratio) + 1024
        guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return []
        }

        // Supply the input block only once. State lives in a class so the @Sendable
        // closure does not mutate captured variables.
        final class FeedState: @unchecked Sendable {
            var consumed = false
            let buffer: AVAudioPCMBuffer
            init(buffer: AVAudioPCMBuffer) {
                self.buffer = buffer
            }
        }
        let feed = FeedState(buffer: monoBuffer)
        var conversionError: NSError? = nil
        let status = converter.convert(to: converted, error: &conversionError) { _, outStatus in
            if feed.consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            feed.consumed = true
            outStatus.pointee = .haveData
            return feed.buffer
        }
        if status == .error {
            if reportedError != true {
                reportedError = true
                print("Warning: audio conversion failed: \(conversionError?.localizedDescription ?? "unknown")")
            }
            return []
        }
        let count = Int(converted.frameLength)
        if count == 0 {
            return []
        }
        guard let outChannel = converted.floatChannelData else {
            return []
        }
        var samples = [Float](repeating: 0.0, count: count)
        samples.withUnsafeMutableBufferPointer { dst in
            guard let base = dst.baseAddress else {
                return
            }
            base.update(from: outChannel[0], count: count)
        }
        return samples
    }
}

/// Holds everything the capture path uses.
///
/// In Swift 6, top-level variables in main.swift are implicitly @MainActor-isolated.
/// The audio tap runs on a real-time thread, so touching them traps in the isolation
/// check (dispatch_assert_queue). Everything the tap uses is an instance property here.
final class MicSession: @unchecked Sendable {
    private let engine: AVAudioEngine
    private let tapFormat: AVAudioFormat
    private let micConverter: MicConverter
    private let transcriber: Transcriber
    private let display: StatusDisplay?

    init(
        engine: AVAudioEngine,
        tapFormat: AVAudioFormat,
        micConverter: MicConverter,
        transcriber: Transcriber,
        display: StatusDisplay?
    ) {
        self.engine = engine
        self.tapFormat = tapFormat
        self.micConverter = micConverter
        self.transcriber = transcriber
        self.display = display
    }

    func start() throws {
        let converterRef = micConverter
        let transcriberRef = transcriber
        engine.inputNode.installTap(
            onBus: 0,
            bufferSize: 4096,
            format: tapFormat
        ) { buffer, _ in
            let samples = converterRef.convert(buffer)
            if samples.isEmpty != true {
                transcriberRef.append(samples)
            }
        }
        try engine.start()
    }

    /// Stops capture, then flushes buffered audio.
    func finish() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        // Do not let a slow recognition block shutdown.
        transcriber.flushRemaining(timeout: 2.0)
        display?.finish()
    }
}

let session = MicSession(
    engine: engine,
    tapFormat: tapFormat,
    micConverter: MicConverter(
        converter: converter,
        monoSourceFormat: monoSourceFormat,
        targetFormat: targetFormat
    ),
    transcriber: transcriber,
    display: statusDisplay
)

shutdown.attach(session)

do {
    try session.start()
} catch {
    print("Error: cannot start the microphone: \(error)")
    print("  Allow the terminal in System Settings > Privacy & Security > Microphone.")
    exit(1)
}

RunLoop.main.run()
