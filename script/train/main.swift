import Foundation
import Haitaka
import MLX
#if canImport(Darwin)
import Darwin
#endif

setbuf(stdout, nil)

print("==================================================")
print("=== haitaka train / eval ===")
print("==================================================")

/// Training and evaluation defaults. Edit these values directly instead of adding command-line options.
enum Defaults {
    /// Longest utterance used for training (40 ms frames, 60 s).
    static let maxTrainFrames = 1500

    static let lrMax: Float = 0.003
    static let lrMin: Float = 0.0005
    /// Upper bound of the cosine decay in steps; longer runs continue at lrMin.
    static let lrDecaySteps = 50_000
    static let warmupStepDivisor = 50
    static let minWarmupSteps = 100
    /// Roll back to the best weights and scale the learning rate when the epoch loss rises this many times in a row.
    static let lossRiseEpochs = 2
    static let lrRollbackFactor: Float = 0.5
    static let maxLRRollbacks = 2

    static let checkpointEvery = 10
    static let batchSize = 64
    /// Seeds weight initialization, batch order and augmentation, so two runs differ only by their data.
    static let seed: UInt64 = 1
    /// Batches within this window are reordered so equal padded lengths run back to back; compiled steps
    /// reallocate GPU buffers whenever the shape changes.
    static let bucketRunWindow = 64
    /// Training utterances evaluated after training, sampled evenly. Held-out utterances are always evaluated in full.
    static let maxTrainEvalSamples = 5000
}

func performanceCoreCount() -> Int {
    var count: Int32 = 0
    var size = MemoryLayout<Int32>.size
    if sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, nil, 0) == 0 && 0 < count {
        return Int(count)
    }
    return ProcessInfo.processInfo.activeProcessorCount
}

var numWorkers = performanceCoreCount()
var epochs = 20
var maxTrainSamples: Int? = nil
var batchSize = Defaults.batchSize
var datasetPath = ""
var englishDictPath = ""
var noiseBankPath = ""
var dictTextPaths: [String] = []
var kanaLMPath = ""
var evalDumpPath = ""
var augmentOptions: Set<String> = []
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-h", "--help":
        print("Usage: train -d <manifest.jsonl> --english-dict <cmudict> [options]")
        print("Options:")
        print("  -d, --dataset <path>               Manifest (JSONL, one {\"path\", \"text\"} per line). The first -s entries are the training set, the rest are eval only [required]")
        print("  -b, --batch-size <Int>             Mini-batch size (default: 64)")
        print("  -e, --epochs <Int>                 Number of epochs (default: 20; 0 = eval only)")
        print("  -s, --samples <Int>                Number of leading entries used for training (default: all)")
        print("  -p, --parallel <Int>               Number of parallel workers (default: P-core count)")
        print("  --export-weights <path>            Output path for the trained weights")
        print("  --import-weights <path>            Initial weights (eval only with -e 0)")
        print("  --english-dict <path>              English pronunciation dictionary (CMU format) [required]")
        print("  --noise-bank <jsonl>               Manifest of noise clips mixed in during training")
        print("  --dict-text <path>                 Text added only to the kana-kanji dictionary (.txt/.srt or a directory; repeatable)")
        print("  --kana-lm <path>                   Kana text, one sentence per line (kanalm output). Fused into the eval CTC beam search as an n-gram LM")
        print("  --eval-dump <jsonl>                Write per-utterance eval results")
        print("  --augment <speed,spec>             Training data augmentation (speed: 0.9-1.1x speed perturbation, spec: SpecAugment)")
        exit(0)
    case "-p", "--parallel":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                numWorkers = max(1, val)
            }
            argIdx += 1
        }
    case "-e", "--epochs":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                epochs = max(0, val)
            }
            argIdx += 1
        }
    case "-s", "--samples":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                maxTrainSamples = max(1, val)
            }
            argIdx += 1
        }
    case "-b", "--batch-size":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                batchSize = max(1, val)
            }
            argIdx += 1
        }
    case "-d", "--dir", "--dataset":
        if (argIdx + 1) < args.count {
            datasetPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--english-dict":
        if (argIdx + 1) < args.count {
            englishDictPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--noise-bank":
        if (argIdx + 1) < args.count {
            noiseBankPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--dict-text":
        if (argIdx + 1) < args.count {
            dictTextPaths.append(args[argIdx + 1])
            argIdx += 1
        }
    case "--kana-lm":
        if (argIdx + 1) < args.count {
            kanaLMPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--eval-dump":
        if (argIdx + 1) < args.count {
            evalDumpPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--augment":
        if (argIdx + 1) < args.count {
            for item in args[argIdx + 1].split(separator: ",") {
                let name = item.trimmingCharacters(in: .whitespaces)
                if name == "speed" || name == "spec" {
                    augmentOptions.insert(name)
                    continue
                }
                if name.isEmpty != true {
                    print("error: unknown --augment value \(name) (expected speed, spec)")
                    exit(1)
                }
            }
            argIdx += 1
        }
    case "--export-weights":
        if (argIdx + 1) < args.count {
            exportWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--import-weights":
        if (argIdx + 1) < args.count {
            importWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    default:
        if arg.hasPrefix("-") != true {
            datasetPath = arg
        }
    }
    argIdx += 1
}

if englishDictPath.isEmpty {
    print("error: specify an English pronunciation dictionary (CMU format) with --english-dict")
    exit(1)
}
guard let englishDict = try? EnglishPronunciations(contentsOfFile: englishDictPath) else {
    print("error: cannot read pronunciation dictionary \(englishDictPath)")
    exit(1)
}
print("English dictionary: \(englishDictPath) (\(englishDict.count) words)")
var noiseBank: NoiseBank? = nil
if noiseBankPath.isEmpty != true {
    guard let bank = try? NoiseBank(manifestPath: noiseBankPath), 0 < bank.count else {
        print("error: cannot read noise bank \(noiseBankPath)")
        exit(1)
    }
    noiseBank = bank
    print("noise bank: \(noiseBankPath) (\(bank.count) clips, \(String(format: "%.1f", bank.totalSeconds / 60.0)) min; probability \(NoiseBank.probability), SNR \(NoiseBank.snrRange.lowerBound)-\(NoiseBank.snrRange.upperBound) dB)")
}
let augmenter = FeatureAugmenter(
    speed: augmentOptions.contains("speed"),
    specAugment: augmentOptions.contains("spec"),
    noiseBank: noiseBank
)
if augmentOptions.isEmpty != true {
    print("augmentation: \(augmentOptions.sorted().joined(separator: ", ")) (speed \(FeatureAugmenter.speedRange.lowerBound)-\(FeatureAugmenter.speedRange.upperBound)x / time masks \(FeatureAugmenter.timeMasks) x <= \(FeatureAugmenter.maxTimeMaskFrames) frames / frequency masks \(FeatureAugmenter.frequencyMasks) x <= \(FeatureAugmenter.maxFrequencyMaskBands) bands)")
}

if datasetPath.isEmpty {
    print("error: specify a manifest (JSONL). See train --help.")
    exit(1)
}

print("dataset: \(datasetPath)")
print("batch size (-b): \(batchSize)")
print("workers    (-p): \(numWorkers) threads")
print("epochs     (-e): \(epochs)")
if let s = maxTrainSamples {
    print("samples    (-s): \(s)")
} else {
    print("samples    (-s): all")
}
if let exp = exportWeightsPath {
    print("export weights: \(exp)")
}
if let imp = importWeightsPath {
    print("import weights: \(imp)")
}

struct ManifestEntry: Codable {
    let path: String
    let text: String
}

guard let manifestContent = try? String(contentsOfFile: datasetPath, encoding: .utf8) else {
    print("error: cannot read manifest \(datasetPath)")
    exit(1)
}

let manifestDir = (datasetPath as NSString).deletingLastPathComponent
let jsonDecoder = JSONDecoder()
var rawPairs: [(path: String, fileId: String, text: String)] = []

// Split on LF only: CharacterSet.newlines would also split on U+2028 inside subtitle text.
for line in manifestContent.components(separatedBy: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty {
        continue
    }
    guard let entry = try? jsonDecoder.decode(ManifestEntry.self, from: Data(trimmed.utf8)) else {
        print("error: cannot parse manifest line: \(trimmed.prefix(120))")
        exit(1)
    }
    var wavPath = entry.path
    if wavPath.hasPrefix("/") != true {
        wavPath = (manifestDir as NSString).appendingPathComponent(wavPath)
    }
    let fileId = ((wavPath as NSString).lastPathComponent as NSString).deletingPathExtension
    rawPairs.append((path: wavPath, fileId: fileId, text: entry.text))
}

// Vocabularies and the kana-kanji dictionary come from the training split only, so held-out transcripts
// never leak into kana-kanji conversion.
let sampleLimit = min(maxTrainSamples ?? rawPairs.count, rawPairs.count)
let trainTextLines = rawPairs.prefix(sampleLimit).map { $0.text }

let kanjiConverter = KanjiConverter(english: englishDict)

var importedWeights: HawkWeights? = nil
if let impPath = importWeightsPath {
    guard let weights = try? HawkWeights.load(from: URL(fileURLWithPath: impPath)) else {
        print("error: cannot read weights file \(impPath)")
        exit(1)
    }
    importedWeights = weights
}
// Weights from pretrain carry no kana vocabulary; their layers start CTC training with a fresh readout.
var pretrainedWeights: HawkWeights? = nil
if let weights = importedWeights, weights.vocabulary == nil {
    if epochs == 0 {
        print("error: pretrained weights have no kana readout to evaluate")
        exit(1)
    }
    pretrainedWeights = weights
    importedWeights = nil
    print("pretrained layers: \(importWeightsPath ?? "")")
}
// Imported weights carry their own kana vocabulary so output IDs stay stable across manifests.
let kanaVocabulary: TextVocabulary
switch importedWeights?.vocabulary {
case .some(let embedded):
    kanaVocabulary = embedded
    print("using embedded kana vocabulary: \(embedded.size) symbols")
case .none:
    kanaVocabulary = TextVocabulary(corpus: trainTextLines.map { kanjiConverter.convertToHiragana($0) })
}

var dictTextLines: [String] = []
for path in dictTextPaths {
    let lines = DictionaryTextCorpus.load(path: path)
    if lines.isEmpty {
        print("error: cannot read dictionary text from \(path)")
        exit(1)
    }
    dictTextLines.append(contentsOf: lines)
}

@MainActor func loadKanaLanguageModel() -> KanaNgramLM? {
    if kanaLMPath.isEmpty {
        return nil
    }
    let lmStart = CFAbsoluteTimeGetCurrent()
    guard let lm = KanaNgramLM(contentsOfFile: kanaLMPath, vocabulary: kanaVocabulary) else {
        print("error: cannot read kana text \(kanaLMPath)")
        exit(1)
    }
    print("kana \(lm.order)-gram LM: weight \(KanaNgramLM.fusionWeight), length bonus \(KanaNgramLM.fusionLengthBonus) (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - lmStart)) s)")
    return lm
}
let kanaLanguageModel = loadKanaLanguageModel()

let kanaKanjiDict = KanaKanjiDictionary()
kanaKanjiDict.buildFromCorpus(rawTexts: trainTextLines, converter: kanjiConverter)
let dictEntriesFromSpeech = kanaKanjiDict.count
if dictTextLines.isEmpty != true {
    kanaKanjiDict.buildFromCorpus(rawTexts: dictTextLines, converter: kanjiConverter)
    print("dictionary text: \(dictTextLines.count) lines (\(dictTextPaths.count) paths) -> dictionary entries \(dictEntriesFromSpeech) -> \(kanaKanjiDict.count)")
}
print("manifest: \(rawPairs.count) entries (training set: \(sampleLimit))")
print("kana vocabulary: \(kanaVocabulary.size) symbols / kana-kanji dictionary: \(kanaKanjiDict.count) entries")

print("\n--- 1. Building dataset (\(sampleLimit) samples) ---")
let startTime = CFAbsoluteTimeGetCurrent()
let dataset = SpeechDataset.lazyFromManifest(
    pairs: rawPairs.prefix(sampleLimit).map { (path: $0.path, text: $0.text) },
    workers: numWorkers,
    english: englishDict
)
print("dataset ready: \(dataset.count) samples (\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - startTime)) s)")
for i in 0..<min(3, dataset.count) {
    let meta = dataset.metaSamples[i]
    print("  [\(i + 1)] \"\(meta.rawText)\"")
    print("      kana: \"\(meta.hiraganaText)\" (\(meta.hiraganaText.count) chars, \(meta.frameCount) frames)")
}

print("\n--- 2. Training the acoustic model ---")
let inputDim = StreamingFeatureFrontEnd.inputDim
print("Hawk: input \(inputDim), layers \(HawkShape.numLayers), width \(HawkShape.modelDim), RNN width \(HawkShape.rnnDim), MLP width \(HawkShape.mlpDim), gate blocks \(HawkShape.heads), conv \(HawkShape.convWidth) frames, lookahead \(HawkShape.convRight) frames/layer")

for weights in [importedWeights, pretrainedWeights].compactMap({ $0 }) {
    guard weights.outputDim == kanaVocabulary.size || weights.vocabulary == nil,
          weights.inputDim == inputDim,
          weights.numLayers == HawkShape.numLayers,
          weights.modelDim == HawkShape.modelDim,
          weights.rnnDim == HawkShape.rnnDim,
          weights.mlpDim == HawkShape.mlpDim,
          weights.heads == HawkShape.heads,
          weights.convWidth == HawkShape.convWidth,
          weights.convRight == HawkShape.convRight || epochs == 0 else {
        print("error: weights do not match Defaults (input \(weights.inputDim)/\(inputDim), output \(weights.outputDim)/\(kanaVocabulary.size), layers \(weights.numLayers)/\(HawkShape.numLayers), width \(weights.modelDim)/\(HawkShape.modelDim), lookahead \(weights.convRight)/\(HawkShape.convRight))")
        exit(1)
    }
}

var network: HawkNetwork? = importedWeights.map { HawkNetwork(weights: $0) }

if 0 < epochs {
    let allTargets: [[Int]] = (0..<dataset.count).map { kanaVocabulary.textToIds(dataset.hiraganaText(at: $0)) }

    MLXRandom.seed(Defaults.seed)
    let mlxNetwork = MLXHawkNetwork(outputDim: kanaVocabulary.size)
    if let weights = importedWeights {
        mlxNetwork.importWeights(from: weights)
    }
    if let weights = pretrainedWeights {
        mlxNetwork.importEncoder(from: weights)
    }
    print("  parameters \(mlxNetwork.exportWeights().parameterCount), batch \(batchSize)")
    let trainer = MLXHawkTrainer(network: mlxNetwork, learningRate: Defaults.lrMax)
    var scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin)
    // Fine-tuning from trained weights anneals from lrMin to zero over the whole run, without warmup.
    if importedWeights != nil {
        scheduler = CosineLRScheduler(lrMax: Defaults.lrMin, lrMin: 0.0)
    }
    print("  learning rate: \(scheduler.lrMax) -> \(scheduler.lrMin)")
    let trainStartTime = CFAbsoluteTimeGetCurrent()

    // CTC needs at least one frame per label plus one per repeated pair.
    func ctcMinimumFrames(_ labels: [Int]) -> Int {
        var required = labels.count
        var i = 1
        while i < labels.count {
            if labels[i] == labels[i - 1] {
                required += 1
            }
            i += 1
        }
        return required
    }
    var feasibleIndices: [Int] = []
    var infeasibleCount = 0
    var tooLongCount = 0
    var di = 0
    while di < dataset.count {
        let frames = dataset.frameCount(at: di)
        if Defaults.maxTrainFrames < frames {
            tooLongCount += 1
            di += 1
            continue
        }
        if ctcMinimumFrames(allTargets[di]) <= frames {
            feasibleIndices.append(di)
        } else {
            infeasibleCount += 1
        }
        di += 1
    }
    if 0 < infeasibleCount {
        print("  excluded as CTC-infeasible: \(infeasibleCount) (frames < required label length)")
    }
    if 0 < tooLongCount {
        print("  excluded as too long: \(tooLongCount) (over \(Defaults.maxTrainFrames) frames)")
    }

    // Batches are formed from length-sorted utterances once; their order is shuffled every epoch.
    let lengthSortedIndices = feasibleIndices.sorted { a, b in
        return dataset.frameCount(at: a) < dataset.frameCount(at: b)
    }
    var batchGroups: [[Int]] = []
    var gStart = 0
    while gStart < lengthSortedIndices.count {
        let gEnd = min(gStart + batchSize, lengthSortedIndices.count)
        batchGroups.append(Array(lengthSortedIndices[gStart..<gEnd]))
        gStart = gEnd
    }

    func paddedFrameCount(of group: [Int]) -> Int {
        var maxFrames = 0
        for idx in group {
            maxFrames = max(maxFrames, dataset.frameCount(at: idx))
        }
        return ((maxFrames + 31) / 32) * 32
    }
    func groupBucketsWithinWindows(_ groups: inout [[Int]], window: Int) {
        var start = 0
        while start < groups.count {
            let end = min(start + window, groups.count)
            let keyed = groups[start..<end].map { group in (key: paddedFrameCount(of: group), group: group) }
            groups.replaceSubrange(start..<end, with: keyed.sorted { a, b in a.key < b.key }.map { $0.group })
            start = end
        }
    }

    final class FeatureBatchBuffer: @unchecked Sendable {
        var items: [[[Float]]]
        init(count: Int) {
            self.items = [[[Float]]](repeating: [], count: count)
        }
    }
    let activeWorkers = numWorkers
    @Sendable func buildBatchFeatures(_ indices: [Int]) -> [[[Float]]] {
        let buffer = FeatureBatchBuffer(count: indices.count)
        let workerCount = max(1, min(activeWorkers, indices.count))
        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            var i = worker
            while i < indices.count {
                let meta = dataset.metaSamples[indices[i]]
                buffer.items[i] = autoreleasepool {
                    SpeechDataset.loadFeatures(
                        path: meta.path,
                        pcmTransform: augmenter.pcmTransform,
                        featureTransform: augmenter.featureTransform
                    ).features
                }
                i += workerCount
            }
        }
        return buffer.items
    }

    final class PrefetchBox: @unchecked Sendable {
        var value: [[[Float]]] = []
    }

    let totalSteps = max(1, epochs * batchGroups.count)
    var warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
    var decaySteps = min(totalSteps - warmupSteps, Defaults.lrDecaySteps)
    if importedWeights != nil {
        warmupSteps = 0
        decaySteps = totalSteps
    }
    let scheduleSteps = warmupSteps + decaySteps
    print("  training steps: \(totalSteps) (warmup \(warmupSteps), decay \(decaySteps), then lrMin)")

    let checkpointInterval = min(Defaults.checkpointEvery, max(1, epochs / 6))
    print("  checkpoint interval: every \(checkpointInterval) epochs")

    var bestLoss = Float.greatestFiniteMagnitude
    var bestEpoch = 0
    var bestWeights: HawkWeights? = nil
    var prevLoss = Float.greatestFiniteMagnitude
    var riseStreak = 0
    var lrScale: Float = 1.0
    var rollbacks = 0

    var globalStep = 0
    var batchOrder = SplitMix64(seed: Defaults.seed)
    var ep = 1
    while ep <= epochs {
        let epStartTime = CFAbsoluteTimeGetCurrent()
        augmenter.epochSalt = UInt64(ep)
        batchGroups.shuffle(using: &batchOrder)
        groupBucketsWithinWindows(&batchGroups, window: Defaults.bucketRunWindow)
        var curLR = scheduler.learningRate(step: globalStep + 1, totalSteps: scheduleSteps, warmupSteps: warmupSteps) * lrScale

        var epLossSum: Float = 0.0
        var batchCount = 0
        var gpuSeconds = 0.0
        var waitSeconds = 0.0
        var bucketSeconds: [Int: (count: Int, seconds: Double)] = [:]
        var bucketSwitches = 0
        var previousPaddedFrames = 0

        var currentBatch: [[[Float]]] = []
        if 0 < batchGroups.count {
            currentBatch = buildBatchFeatures(batchGroups[0])
        }
        var bIdx = 0
        while bIdx < batchGroups.count {
            let prefetchBox = PrefetchBox()
            let prefetchGroup = DispatchGroup()
            if (bIdx + 1) < batchGroups.count {
                let nextIndices = batchGroups[bIdx + 1]
                prefetchGroup.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    prefetchBox.value = buildBatchFeatures(nextIndices)
                    prefetchGroup.leave()
                }
            }

            let targets = batchGroups[bIdx].map { allTargets[$0] }
            globalStep += 1
            curLR = scheduler.learningRate(step: globalStep, totalSteps: scheduleSteps, warmupSteps: warmupSteps) * lrScale
            trainer.updateLearningRate(curLR)

            var maxFrames = 0
            for seq in currentBatch {
                maxFrames = max(maxFrames, seq.count)
            }
            let paddedFrames = ((maxFrames + 31) / 32) * 32
            let gpuStart = CFAbsoluteTimeGetCurrent()
            let loss = trainer.trainBatchCTC(featuresBatch: currentBatch, targetsBatch: targets, blankId: TextVocabulary.padId)
            if trainer.lastGradientNorm.isFinite != true || loss.isFinite != true {
                print("    ⚠ loss \(loss) / grad norm \(trainer.lastGradientNorm) (step \(globalStep), \(paddedFrames) frames)")
            }
            let gpuElapsed = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuSeconds += gpuElapsed
            let prev = bucketSeconds[paddedFrames] ?? (count: 0, seconds: 0.0)
            bucketSeconds[paddedFrames] = (count: prev.count + 1, seconds: prev.seconds + gpuElapsed)
            if paddedFrames != previousPaddedFrames {
                bucketSwitches += 1
                previousPaddedFrames = paddedFrames
            }
            epLossSum += loss
            batchCount += 1

            let waitStart = CFAbsoluteTimeGetCurrent()
            prefetchGroup.wait()
            waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
            currentBatch = prefetchBox.value
            bIdx += 1
        }

        let avgLoss = epLossSum / Float(max(1, batchCount))
        let epElapsed = CFAbsoluteTimeGetCurrent() - epStartTime
        print("  Epoch [\(ep)/\(epochs)] - loss: \(String(format: "%.4f", avgLoss)) (LR: \(String(format: "%.5f", curLR)), time: \(String(format: "%.2f", epElapsed)) s)")
        print("    breakdown: GPU \(String(format: "%.0f", gpuSeconds)) s / prefetch wait \(String(format: "%.0f", waitSeconds)) s / \(batchCount) batches / \(bucketSwitches) bucket switches")
        print("    MLX memory: active \(MLX.Memory.activeMemory >> 20) MB / cache \(MLX.Memory.cacheMemory >> 20) MB / peak \(MLX.Memory.peakMemory >> 20) MB")
        let topBuckets = bucketSeconds.sorted { a, b in b.value.seconds < a.value.seconds }.prefix(6)
        var bucketLine = "    length buckets (frames: batches / s):"
        for (frames, stat) in topBuckets {
            bucketLine += " \(frames): \(stat.count) / \(String(format: "%.0f", stat.seconds))"
        }
        print(bucketLine)

        if avgLoss < bestLoss {
            bestLoss = avgLoss
            bestEpoch = ep
            bestWeights = mlxNetwork.exportWeights(vocabulary: kanaVocabulary)
        }
        if prevLoss < avgLoss {
            riseStreak += 1
        } else {
            riseStreak = 0
        }
        prevLoss = avgLoss

        if Defaults.lossRiseEpochs <= riseStreak {
            guard let best = bestWeights, rollbacks < Defaults.maxLRRollbacks else {
                print("    ■ loss rose for \(Defaults.lossRiseEpochs) epochs with no rollbacks left; stopping (best epoch \(bestEpoch))")
                break
            }
            rollbacks += 1
            lrScale *= Defaults.lrRollbackFactor
            riseStreak = 0
            prevLoss = bestLoss
            trainer.rollback(to: best)
            print("    ↩ loss rose for \(Defaults.lossRiseEpochs) epochs; rolling back to epoch \(bestEpoch) (loss \(String(format: "%.4f", bestLoss))) and scaling the learning rate by \(String(format: "%.2f", lrScale)) (\(rollbacks)/\(Defaults.maxLRRollbacks))")
        }

        if let basePath = exportWeightsPath, (ep % checkpointInterval) == 0 && ep < epochs {
            let ckptPath = "\(basePath).ep\(ep).json"
            do {
                try mlxNetwork.exportWeights(vocabulary: kanaVocabulary).save(to: URL(fileURLWithPath: ckptPath))
                print("    ✓ checkpoint saved: \(ckptPath)")
            } catch {
                print("    ✕ failed to save checkpoint: \(error)")
            }
        }
        ep += 1
    }

    if let best = bestWeights, bestLoss < prevLoss {
        mlxNetwork.importWeights(from: best)
        print("  epoch \(bestEpoch) (loss \(String(format: "%.4f", bestLoss))) beats the final epoch; using its weights")
    }
    print("\nGPU training done (total \(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - trainStartTime)) s)")
    network = HawkNetwork(weights: mlxNetwork.exportWeights(vocabulary: kanaVocabulary))
    MLX.Memory.clearCache()
}

guard let model = network else {
    print("error: no weights to evaluate (-e 0 requires --import-weights)")
    exit(1)
}

if let expPath = exportWeightsPath {
    do {
        try model.weights.save(to: URL(fileURLWithPath: expPath))
        print("\n✓ weights written: \(expPath)")
    } catch {
        print("\n✕ failed to save weights: \(error)")
    }
}

print("\n==================================================")
print("=== 3. Evaluation ===")
print("==================================================")

@Sendable func stripPunctuation(_ text: String) -> String {
    var out = ""
    for c in text {
        switch c {
        case "、", "。", "，", "．", "・", "？", "！", "?", "!":
            break
        default:
            out.append(c)
        }
    }
    return out
}

@Sendable func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
    let a1 = Array(s1)
    let a2 = Array(s2)
    let m = a1.count
    let n = a2.count
    if m == 0 {
        return n
    }
    if n == 0 {
        return m
    }
    var prev = Array(0...n)
    var curr = [Int](repeating: 0, count: n + 1)
    var i = 1
    while i <= m {
        curr[0] = i
        var j = 1
        while j <= n {
            if a1[i - 1] == a2[j - 1] {
                curr[j] = prev[j - 1]
            } else {
                curr[j] = 1 + min(prev[j - 1], min(prev[j], curr[j - 1]))
            }
            j += 1
        }
        swap(&prev, &curr)
        i += 1
    }
    return prev[n]
}

struct EvalResult {
    let fileId: String
    /// Parent directory of the WAV file, used for per-corpus breakdowns.
    let corpus: String
    let targetText: String
    let predText: String
    let cer: Float
    let isTrain: Bool
    let targetKana: String
    let predKana: String
    let kanaCer: Float
    /// Kana-kanji conversion alone: kanji CER when the reference kana is fed to the kana-kanji decoder.
    let goldKanji: String
    let goldKanjiCer: Float
    let goldKanjiCerNoPunct: Float
    let goldExactNoPunct: Bool
}

struct GroupSummary {
    let count: Int
    let exactRate: Float
    let meanCer: Float
    let medianCer: Float
    let meanKanaCer: Float
    let medianKanaCer: Float
    let meanGoldKanjiCer: Float
    let goldExactRate: Float
    let meanGoldKanjiCerNoPunct: Float
    let goldExactRateNoPunct: Float
}

func computeSummary(_ results: [EvalResult]) -> GroupSummary {
    let n = results.count
    if n == 0 {
        return GroupSummary(count: 0, exactRate: 0, meanCer: 0, medianCer: 0, meanKanaCer: 0, medianKanaCer: 0,
                            meanGoldKanjiCer: 0, goldExactRate: 0, meanGoldKanjiCerNoPunct: 0, goldExactRateNoPunct: 0)
    }
    var exact = 0
    var goldExact = 0
    var goldExactNoPunct = 0
    var sumGoldCer: Float = 0.0
    var sumGoldCerNoPunct: Float = 0.0
    for r in results {
        if r.targetText == r.predText {
            exact += 1
        }
        if r.goldKanji == r.targetText {
            goldExact += 1
        }
        if r.goldExactNoPunct {
            goldExactNoPunct += 1
        }
        sumGoldCer += r.goldKanjiCer
        sumGoldCerNoPunct += r.goldKanjiCerNoPunct
    }
    let cers = results.map { $0.cer }.sorted()
    let kanaCers = results.map { $0.kanaCer }.sorted()
    return GroupSummary(
        count: n,
        exactRate: Float(exact) * 100.0 / Float(n),
        meanCer: cers.reduce(0, +) / Float(n) * 100.0,
        medianCer: cers[n / 2] * 100.0,
        meanKanaCer: kanaCers.reduce(0, +) / Float(n) * 100.0,
        medianKanaCer: kanaCers[n / 2] * 100.0,
        meanGoldKanjiCer: sumGoldCer / Float(n) * 100.0,
        goldExactRate: Float(goldExact) * 100.0 / Float(n),
        meanGoldKanjiCerNoPunct: sumGoldCerNoPunct / Float(n) * 100.0,
        goldExactRateNoPunct: Float(goldExactNoPunct) * 100.0 / Float(n)
    )
}

final class EvalBuffer: @unchecked Sendable {
    var results: [EvalResult?]
    init(count: Int) {
        self.results = [EvalResult?](repeating: nil, count: count)
    }
}

// Training utterances are sampled evenly across the split; held-out utterances are all evaluated.
var evalIndices: [Int] = []
let trainEvalTarget = min(Defaults.maxTrainEvalSamples, sampleLimit)
var previousBucket = -1
var selectIdx = 0
while selectIdx < sampleLimit {
    let bucket = (selectIdx * trainEvalTarget) / max(1, sampleLimit)
    if bucket != previousBucket {
        evalIndices.append(selectIdx)
        previousBucket = bucket
    }
    selectIdx += 1
}
evalIndices += Array(sampleLimit..<rawPairs.count)

let evalPairs = evalIndices.map { rawPairs[$0] }
let evalIsTrain = evalIndices.map { $0 < sampleLimit }
let evalBuffer = EvalBuffer(count: evalPairs.count)
let evalWorkers = max(1, numWorkers)
print("eval targets: seen set \(trainEvalTarget) + unseen set \(rawPairs.count - sampleLimit)")
let evalStartTime = Date()
DispatchQueue.concurrentPerform(iterations: evalWorkers) { worker in
    let converter = KanjiConverter(english: englishDict)
    let ctcDecoder = CTCDecoder(vocabulary: kanaVocabulary, languageModel: kanaLanguageModel)
    let kanjiDecoder = KanaKanjiDecoder(dictionary: kanaKanjiDict)
    var idx = worker
    while idx < evalPairs.count {
        let pair = evalPairs[idx]
        defer { idx += evalWorkers }
        guard let wavData = SpeechDataset.loadWavFile(path: pair.path) else {
            continue
        }
        let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
        let features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k)
        let kana = ctcDecoder.decode(logProbs: model.logProbs(features: features)).text
        let kanji = kanjiDecoder.decode(kanaText: kana)
        let targetKana = converter.convertToHiragana(pair.text)
        let goldKanji = kanjiDecoder.decode(kanaText: targetKana)
        let targetNoPunct = stripPunctuation(pair.text)
        let goldNoPunct = stripPunctuation(goldKanji)
        // Kana CER is scored without punctuation so it stays comparable with models that never emit it.
        let targetKanaScored = stripPunctuation(targetKana)
        let kanaScored = stripPunctuation(kana)

        var corpusDir = ((pair.path as NSString).deletingLastPathComponent as NSString).lastPathComponent
        if corpusDir == "wav" {
            let parent = ((pair.path as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent
            corpusDir = (parent as NSString).lastPathComponent
        }
        let textLength = Float(max(1, pair.text.count))
        evalBuffer.results[idx] = EvalResult(
            fileId: pair.fileId,
            corpus: corpusDir,
            targetText: pair.text,
            predText: kanji,
            cer: Float(levenshteinDistance(pair.text, kanji)) / textLength,
            isTrain: evalIsTrain[idx],
            targetKana: targetKana,
            predKana: kana,
            kanaCer: Float(levenshteinDistance(targetKanaScored, kanaScored)) / Float(max(1, targetKanaScored.count)),
            goldKanji: goldKanji,
            goldKanjiCer: Float(levenshteinDistance(pair.text, goldKanji)) / textLength,
            goldKanjiCerNoPunct: Float(levenshteinDistance(targetNoPunct, goldNoPunct)) / Float(max(1, targetNoPunct.count)),
            goldExactNoPunct: targetNoPunct == goldNoPunct
        )
    }
}
print(String(format: "eval time: %.1f s (%d workers)", Date().timeIntervalSince(evalStartTime), evalWorkers))

let allEval = evalBuffer.results.compactMap { $0 }
let trainResults = allEval.filter { $0.isTrain }
let unseenResults = allEval.filter { $0.isTrain != true }
let trainSummary = computeSummary(trainResults)
let unseenSummary = computeSummary(unseenResults)

/// Articulation class of a kana, for per-class error rates.
func kanaArticulationClass(_ ch: Character) -> String {
    let voicelessPlosive: Set<Character> = ["か", "き", "く", "け", "こ", "た", "ち", "つ", "て", "と", "ぱ", "ぴ", "ぷ", "ぺ", "ぽ"]
    let fricative: Set<Character> = ["さ", "し", "す", "せ", "そ", "は", "ひ", "ふ", "へ", "ほ"]
    let voicedObstruent: Set<Character> = [
        "が", "ぎ", "ぐ", "げ", "ご", "だ", "ぢ", "づ", "で", "ど",
        "ば", "び", "ぶ", "べ", "ぼ", "ざ", "じ", "ず", "ぜ", "ぞ"
    ]
    let nasal: Set<Character> = ["な", "に", "ぬ", "ね", "の", "ま", "み", "む", "め", "も", "ん"]
    let approximant: Set<Character> = ["や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を"]
    let vowel: Set<Character> = ["あ", "い", "う", "え", "お"]
    let special: Set<Character> = ["っ", "ー", "ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゔ"]
    if voicelessPlosive.contains(ch) {
        return "voiceless plosive (k/t/p)"
    }
    if fricative.contains(ch) {
        return "fricative (s/h)"
    }
    if voicedObstruent.contains(ch) {
        return "voiced obstruent (g/d/b/z)"
    }
    if nasal.contains(ch) {
        return "nasal (n/m/N)"
    }
    if approximant.contains(ch) {
        return "approximant (y/r/w)"
    }
    if vowel.contains(ch) {
        return "vowel"
    }
    if special.contains(ch) {
        return "special (geminate/long vowel/small kana)"
    }
    return "other"
}

/// Aligns reference and hypothesis kana and counts correct, substituted and deleted characters per class.
func accumulateKanaErrors(
    target: String,
    pred: String,
    correct: inout [String: Int],
    substituted: inout [String: Int],
    deleted: inout [String: Int],
    insertedTotal: inout Int
) {
    let a = Array(target)
    let b = Array(pred)
    let m = a.count
    let n = b.count
    var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
    var i = 0
    while i <= m {
        dp[i][0] = i
        i += 1
    }
    var j = 0
    while j <= n {
        dp[0][j] = j
        j += 1
    }
    i = 1
    while i <= m {
        j = 1
        while j <= n {
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            dp[i][j] = min(dp[i - 1][j] + 1, min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost))
            j += 1
        }
        i += 1
    }
    i = m
    j = n
    while 0 < i || 0 < j {
        if 0 < i && 0 < j {
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                let cls = kanaArticulationClass(a[i - 1])
                if cost == 0 {
                    correct[cls, default: 0] += 1
                } else {
                    substituted[cls, default: 0] += 1
                }
                i -= 1
                j -= 1
                continue
            }
        }
        if 0 < i && dp[i][j] == dp[i - 1][j] + 1 {
            deleted[kanaArticulationClass(a[i - 1]), default: 0] += 1
            i -= 1
            continue
        }
        insertedTotal += 1
        j -= 1
    }
}

func printKanaErrorAnalysis(_ label: String, _ results: [EvalResult]) {
    var correct: [String: Int] = [:]
    var substituted: [String: Int] = [:]
    var deleted: [String: Int] = [:]
    var insertedTotal = 0
    for r in results {
        accumulateKanaErrors(target: r.targetKana, pred: r.predKana, correct: &correct, substituted: &substituted, deleted: &deleted, insertedTotal: &insertedTotal)
    }
    print("\n[\(label)] accuracy by articulatory class (total insertions: \(insertedTotal))")
    let classOrder = [
        "voiceless plosive (k/t/p)", "fricative (s/h)", "voiced obstruent (g/d/b/z)",
        "nasal (n/m/N)", "approximant (y/r/w)", "vowel", "special (geminate/long vowel/small kana)", "other"
    ]
    for cls in classOrder {
        let c = correct[cls] ?? 0
        let sub = substituted[cls] ?? 0
        let del = deleted[cls] ?? 0
        let total = c + sub + del
        if total == 0 {
            continue
        }
        print("  \(cls): correct \(String(format: "%.1f", Float(c) * 100.0 / Float(total)))% (correct \(c) / sub \(sub) / del \(del), total \(total))")
    }
}

// Latency is measured single-threaded; the parallel evaluation above contends for cores.
if 0 < rawPairs.count {
    var benchFeatures: [[[Float]]] = []
    var benchAudioSeconds: Double = 0.0
    for pair in rawPairs.prefix(20) {
        if let wavData = SpeechDataset.loadWavFile(path: pair.path) {
            let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
            benchAudioSeconds += Double(pcm16k.count) / 16000.0
            benchFeatures.append(SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k))
        }
    }
    print("\n==================================================")
    print("=== [Inference latency] single thread, \(benchFeatures.count) utterances (\(String(format: "%.1f", benchAudioSeconds)) s of audio) ===")
    print("==================================================")
    let decoder = CTCDecoder(vocabulary: kanaVocabulary)
    let acousticStart = CFAbsoluteTimeGetCurrent()
    var totalFrames = 0
    var benchLogProbs: [[[Float]]] = []
    for feats in benchFeatures {
        let logProbs = model.logProbs(features: feats)
        totalFrames += logProbs.count
        benchLogProbs.append(logProbs)
    }
    let acousticElapsed = CFAbsoluteTimeGetCurrent() - acousticStart
    let decodeStart = CFAbsoluteTimeGetCurrent()
    var decodedChars = 0
    for logProbs in benchLogProbs {
        decodedChars += decoder.decode(logProbs: logProbs).text.count
    }
    let fullElapsed = acousticElapsed + (CFAbsoluteTimeGetCurrent() - decodeStart)
    let n = Double(max(1, benchFeatures.count))
    print("[Hawk \(model.weights.numLayers) layers, width \(model.weights.modelDim)]")
    print("  acoustic model only: \(String(format: "%.2f", (acousticElapsed / n) * 1000.0)) ms/utt (\(totalFrames) frames)")
    print("  full kana transcription (acoustic model + CTC beam): \(String(format: "%.2f", (fullElapsed / n) * 1000.0)) ms/utt (\(decodedChars) chars)")
    print("  RTF: \(String(format: "%.4f", fullElapsed / max(1e-9, benchAudioSeconds))) (\(String(format: "%.0f", max(1e-9, benchAudioSeconds) / fullElapsed))x real time)")
}

print("\n==================================================")
print("=== [Kana error analysis] ===")
print("==================================================")
printKanaErrorAnalysis("seen set", trainResults)
printKanaErrorAnalysis("unseen set", unseenResults)

print("\n==================================================")
print("=== [Summary] seen set (\(trainSummary.count)) vs unseen set (\(unseenSummary.count)) ===")
print("=== kana CER = acoustic model / kanji CER = after kana-kanji conversion ===")
print("==================================================")
func summaryLine(_ name: String, _ s: GroupSummary) -> String {
    return "  • \(name) (\(s.count)): Exact: \(String(format: "%.1f", s.exactRate))%, kanji CER: \(String(format: "%.2f", s.meanCer))% (median \(String(format: "%.2f", s.medianCer))%), kana CER: \(String(format: "%.2f", s.meanKanaCer))% (median \(String(format: "%.2f", s.medianKanaCer))%)"
}
print(summaryLine("seen set", trainSummary))
print(summaryLine("unseen set", unseenSummary))
print("  ── kana-kanji conversion alone (kanji CER from reference kana) ──")
print("     with punctuation     seen: \(String(format: "%.2f", trainSummary.meanGoldKanjiCer))% (exact \(String(format: "%.1f", trainSummary.goldExactRate))%) / unseen: \(String(format: "%.2f", unseenSummary.meanGoldKanjiCer))% (exact \(String(format: "%.1f", unseenSummary.goldExactRate))%)")
print("     without punctuation  seen: \(String(format: "%.2f", trainSummary.meanGoldKanjiCerNoPunct))% (exact \(String(format: "%.1f", trainSummary.goldExactRateNoPunct))%) / unseen: \(String(format: "%.2f", unseenSummary.meanGoldKanjiCerNoPunct))% (exact \(String(format: "%.1f", unseenSummary.goldExactRateNoPunct))%)")

let corpusGroups = Dictionary(grouping: allEval, by: { $0.corpus })
if 1 < corpusGroups.count {
    print("\n=== [Per-corpus breakdown] ===")
    for (name, results) in corpusGroups.sorted(by: { $0.key < $1.key }) {
        print(summaryLine(name, computeSummary(results)))
    }
}

func printExamples(_ title: String, _ list: [EvalResult]) {
    print("\n--- [\(title)] ---")
    for (i, r) in list.enumerated() {
        print("  [\(i + 1)] \(r.fileId) (kanji CER: \(String(format: "%.1f", r.cer * 100.0))%, kana CER: \(String(format: "%.1f", r.kanaCer * 100.0))%)")
        print("      ref kana:  \"\(r.targetKana)\"")
        print("      hyp kana:  \"\(r.predKana)\"")
        print("      ref kanji: \"\(r.targetText)\"")
        print("      hyp kanji: \"\(r.predText)\"")
    }
}
let sortedUnseen = unseenResults.sorted { $0.cer < $1.cer }
printExamples("unseen set: best 5", Array(sortedUnseen.prefix(5)))
printExamples("unseen set: worst 5", Array(sortedUnseen.suffix(5).reversed()))

if evalDumpPath.isEmpty != true {
    var lines: [String] = []
    for r in allEval {
        let row: [String: Any] = [
            "fileId": r.fileId,
            "corpus": r.corpus,
            "isTrain": r.isTrain,
            "cer": r.cer,
            "kanaCer": r.kanaCer,
            "targetKana": r.targetKana,
            "predKana": r.predKana,
            "targetKanaLength": r.targetKana.count,
            "predKanaLength": r.predKana.count,
            "targetText": r.targetText,
            "predText": r.predText,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: row, options: [.withoutEscapingSlashes]),
           let line = String(data: data, encoding: .utf8) {
            lines.append(line)
        }
    }
    do {
        try (lines.joined(separator: "\n") + "\n").write(toFile: evalDumpPath, atomically: true, encoding: .utf8)
        print("\nper-utterance eval results written: \(evalDumpPath) (\(lines.count))")
    } catch {
        print("\n✕ cannot write per-utterance eval results: \(evalDumpPath)")
    }
}

if unseenResults.count <= 200 {
    print("\n--- [unseen set, all] fileId kanaCER kanjiCER ---")
    for r in unseenResults {
        print("  \(r.fileId)\t\(String(format: "%.1f", r.kanaCer * 100.0))\t\(String(format: "%.1f", r.cer * 100.0))")
    }
}

print("\n==================================================")
print("=== Training and evaluation complete ===")
print("==================================================")
