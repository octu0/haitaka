import Foundation
import Haitaka

setbuf(stdout, nil)

var wavPath = ""
var weightsPath = ""
var dictPath = ""
var kanaLMPath = ""
var chunkSeconds = 0.0
var maxSeconds = 0.0
var usePostProcess = true

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-i", "--input":
        if (argIdx + 1) < args.count {
            wavPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-d", "--dictionary":
        if (argIdx + 1) < args.count {
            dictPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--kana-lm":
        if (argIdx + 1) < args.count {
            kanaLMPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--chunk-seconds":
        if (argIdx + 1) < args.count {
            if let v = Double(args[argIdx + 1]) {
                chunkSeconds = max(0.0, v)
            }
            argIdx += 1
        }
    case "--max-seconds":
        if (argIdx + 1) < args.count {
            if let v = Double(args[argIdx + 1]) {
                maxSeconds = max(0.0, v)
            }
            argIdx += 1
        }
    case "--raw", "--no-postprocess":
        usePostProcess = false
    default:
        break
    }
    argIdx += 1
}

if wavPath.isEmpty || weightsPath.isEmpty {
    print("Usage: transcribe -i <audio.wav> -w <weights.safetensors> [-d <dictionary text>] [--kana-lm <kana text>] [--chunk-seconds N] [--max-seconds N] [--raw]")
    exit(1)
}

/// Resident memory size in MB.
func residentMemoryMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { infoPtr in
        infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
        }
    }
    if result != KERN_SUCCESS {
        return 0.0
    }
    return Double(info.resident_size) / (1024.0 * 1024.0)
}

func report(_ label: String) {
    print(String(format: "  [memory] %@: %.0f MB", label, residentMemoryMB()))
}

print("==================================================")
print("=== Transcription ===")
print("==================================================")

let loadStart = CFAbsoluteTimeGetCurrent()
guard let wavData = SpeechDataset.loadWavFile(path: wavPath) else {
    print("Error: cannot read audio file: \(wavPath)")
    exit(1)
}
var pcm = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
if 0.0 < maxSeconds {
    let limit = Int(maxSeconds * 16000.0)
    if limit < pcm.count {
        pcm = Array(pcm[0..<limit])
    }
}
let audioSeconds = Double(pcm.count) / 16000.0
print(String(format: "Audio: %.1f min (%.0f s), loaded in %.1f s",
             audioSeconds / 60.0, audioSeconds, CFAbsoluteTimeGetCurrent() - loadStart))
report("after audio load")

guard let weights = try? HawkWeights.load(from: URL(fileURLWithPath: weightsPath)), let kanaVocabulary = weights.vocabulary else {
    print("Error: cannot load weights (they must embed the kana vocabulary): \(weightsPath)")
    exit(1)
}
print("Acoustic model: Hawk, \(weights.numLayers) layers, width \(weights.modelDim), lookahead \(weights.lookaheadFrames * 40) ms, kana vocabulary \(kanaVocabulary.size) chars")

var kanaLanguageModel: KanaNgramLM? = nil
if kanaLMPath.isEmpty != true {
    guard let lm = KanaNgramLM(contentsOfFile: kanaLMPath, vocabulary: kanaVocabulary) else {
        print("Error: cannot read kana text: \(kanaLMPath)")
        exit(1)
    }
    kanaLanguageModel = lm
    print("Kana \(lm.order)-gram LM: \(kanaLMPath)")
}

let kanaKanjiDict = KanaKanjiDictionary()
if dictPath.isEmpty != true {
    guard let dictContent = try? String(contentsOfFile: dictPath, encoding: .utf8) else {
        print("Error: cannot read text: \(dictPath)")
        exit(1)
    }
    let lines = dictContent.components(separatedBy: "\n")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { $0.isEmpty != true }
    kanaKanjiDict.buildFromCorpus(rawTexts: lines)
    print("Kana-kanji dictionary: \(kanaKanjiDict.count) entries")
    report("after dictionary build")
} else {
    print("Kana-kanji dictionary: none (kana output only)")
}

// Split at VAD silences with --chunk-seconds as the per-segment cap (fixed-length cuts
// would split words). 0 disables splitting.
var segments: [(start: Int, count: Int)] = []
if chunkSeconds <= 0.0 {
    segments.append((start: 0, count: pcm.count))
} else {
    let chunkStart = CFAbsoluteTimeGetCurrent()
    let chunker = SpeechChunker(maxSegmentSeconds: Float(chunkSeconds))
    for span in chunker.chunk(pcm: pcm) {
        segments.append((start: span.start, count: span.end - span.start))
    }
    print(String(format: "VAD chunking: %.1f s", CFAbsoluteTimeGetCurrent() - chunkStart))
}
var chunkDescription = " (split at silence, at most \(Int(chunkSeconds)) s)"
if chunkSeconds <= 0.0 {
    chunkDescription = " (no split)"
}
print("Chunks: \(segments.count)" + chunkDescription)

let network = HawkNetwork(weights: weights)
let ctcDecoder = CTCDecoder(vocabulary: kanaVocabulary, languageModel: kanaLanguageModel)
report("after model load")

var kanaParts: [String] = []
var totalFrames = 0
var peakMemory = residentMemoryMB()
let inferStart = CFAbsoluteTimeGetCurrent()

var segIdx = 0
while segIdx < segments.count {
    let seg = segments[segIdx]
    let features = SpeechDataset.extractFeaturesFromPCM(pcmData: Array(pcm[seg.start..<(seg.start + seg.count)]))
    totalFrames += features.count
    if 0 < features.count {
        kanaParts.append(ctcDecoder.decode(logProbs: network.logProbs(features: features)).text)
    } else {
        kanaParts.append("")
    }

    let mem = residentMemoryMB()
    if peakMemory < mem {
        peakMemory = mem
    }
    if (segIdx % 10) == 0 || segIdx == (segments.count - 1) {
        let done = Double(seg.start + seg.count) / 16000.0
        let elapsed = CFAbsoluteTimeGetCurrent() - inferStart
        print(String(format: "  [%d/%d] %.0f s processed / %.1f s elapsed / memory %.0f MB",
                     segIdx + 1, segments.count, done, elapsed, mem))
    }
    segIdx += 1
}

let inferElapsed = CFAbsoluteTimeGetCurrent() - inferStart
let kanaText = kanaParts.joined()
print("")
print("==================================================")
print(String(format: "Kana done: %.1f s (audio %.0f s, RTF %.4f, %.0fx real time)",
             inferElapsed, audioSeconds, inferElapsed / audioSeconds, audioSeconds / inferElapsed))
print("  Frames: \(totalFrames), kana chars: \(kanaText.count)")
print(String(format: "  Peak memory: %.0f MB", peakMemory))

// Second stage: kana-to-kanji conversion, only when a dictionary is loaded.
if 0 < kanaText.count {
    var outputParts = kanaParts
    if 0 < kanaKanjiDict.count {
        let stage2Start = CFAbsoluteTimeGetCurrent()
        let kanaDecoder = KanaKanjiDecoder(dictionary: kanaKanjiDict)
        var kanjiParts: [String] = []
        kanjiParts.reserveCapacity(kanaParts.count)
        for part in kanaParts {
            kanjiParts.append(kanaDecoder.decode(kanaText: part))
        }
        outputParts = kanjiParts
        let stage2Elapsed = CFAbsoluteTimeGetCurrent() - stage2Start
        let totalKanjiChars = kanjiParts.reduce(0) { $0 + $1.count }
        print(String(format: "Kanji done: %.1f s (%d chars)", stage2Elapsed, totalKanjiChars))
        let mem = residentMemoryMB()
        if peakMemory < mem {
            peakMemory = mem
        }
        print(String(format: "  Peak memory: %.0f MB", peakMemory))
        print("")
    }

    if usePostProcess {
        print("=== Formatted output (filler removal, ITN, paragraph timestamps) ===")
        let fillerFilter = FillerWordFilter()
        let normalizer = InverseTextNormalizer()
        var results: [TranscriptionResult] = []
        results.reserveCapacity(segments.count)
        for (idx, seg) in segments.enumerated() {
            let sSec = Float(seg.start) / 16000.0
            let eSec = Float(seg.start + seg.count) / 16000.0
            var text = outputParts[idx]
            text = fillerFilter.filter(text)
            text = normalizer.normalize(text)
            results.append(TranscriptionResult(text: text, startTimeSeconds: sSec, endTimeSeconds: eSec))
        }
        let paragraphs = ParagraphSegmenter.segment(results: results)
        for p in paragraphs {
            print(p.formattedLine)
        }
        print("==================================================")
    } else {
        if 0 < kanaKanjiDict.count {
            print("--- First 300 chars ---")
            print(String(outputParts.joined().prefix(300)))
        }
    }
}
print("")
print("--- First 200 kana chars ---")
print(String(kanaText.prefix(200)))
