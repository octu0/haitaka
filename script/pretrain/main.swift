import Foundation
import Haitaka
import MLX

// HuBERT-style self-supervised pretraining on unlabeled audio. The exported weights go to
// train --import-weights, which keeps the layers and starts CTC with a fresh readout.
//   1st iteration: classes are k-means clusters of the input features.
//     pretrain -d <manifest.jsonl> --kmeans <codebook.json>
//     pretrain -d <manifest.jsonl> --targets <codebook.json> -e <epochs> --export-weights <weights.json>
//   2nd iteration: classes are k-means clusters of a pretrained model's hidden layer (--teacher).
//     pretrain -d <manifest.jsonl> --kmeans <codebook.json> --teacher <weights.json>
//     pretrain -d <manifest.jsonl> --targets <codebook.json> --teacher <weights.json> -e <epochs> --export-weights <weights.json>

setvbuf(stdout, nil, _IOLBF, 0)

enum Defaults {
    /// Longer utterances are split into consecutive windows of this many frames (20.48 s).
    static let windowFrames = 512
    static let minFrames = 16
    static let featureClusters = 256
    static let teacherClusters = 500
    /// The teacher's residual stream after this many layers (6 of 8).
    static let teacherLayers = 6
    static let kmeansFrames = 200_000
    static let kmeansFramesPerFile = 50
    static let kmeansIterations = 25
    /// Each frame starts a masked span with this probability; about 30% of frames end up masked.
    static let maskStartProbability = 0.065
    static let maskSpan = 5
    static let lrMax: Float = 0.003
    static let lrMin: Float = 0.0005
    static let warmupStepDivisor = 50
    static let minWarmupSteps = 100
    static let batchSize = 64
    static let logEvery = 200
}

var manifestPath = ""
var exportPath = ""
var kmeansPath = ""
var targetsPath = ""
var teacherPath = ""
var epochs = 1
var batchSize = Defaults.batchSize
var workers = 8
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    var value = ""
    if (argIdx + 1) < args.count {
        value = args[argIdx + 1]
    }
    switch args[argIdx] {
    case "-d":
        manifestPath = value
    case "-e":
        epochs = max(1, Int(value) ?? epochs)
    case "-b":
        batchSize = max(1, Int(value) ?? batchSize)
    case "-p":
        workers = max(1, Int(value) ?? workers)
    case "--export-weights":
        exportPath = value
    case "--kmeans":
        kmeansPath = value
    case "--targets":
        targetsPath = value
    case "--teacher":
        teacherPath = value
    default:
        print("unknown option: \(args[argIdx])")
        exit(1)
    }
    argIdx += 2
}
if manifestPath.isEmpty || (kmeansPath.isEmpty && (targetsPath.isEmpty || exportPath.isEmpty)) {
    print("Usage: pretrain -d <manifest.jsonl> --kmeans <codebook.json> [--teacher <weights.json>]")
    print("       pretrain -d <manifest.jsonl> --targets <codebook.json> -e <epochs> --export-weights <weights.json> [--teacher <weights.json>] [-b 64] [-p 8]")
    exit(1)
}

struct Row: Decodable {
    let path: String
}

/// Frames [start, start + length) of utterance `path`.
struct Window: Sendable {
    let path: Int
    let start: Int
    let length: Int
}

guard let content = try? String(contentsOfFile: manifestPath, encoding: .utf8) else {
    print("error: cannot read manifest: \(manifestPath)")
    exit(1)
}
var paths: [String] = []
var windows: [Window] = []
for line in content.split(separator: "\n") {
    guard let row = try? JSONDecoder().decode(Row.self, from: Data(line.utf8)),
          let attrs = try? FileManager.default.attributesOfItem(atPath: row.path),
          let size = attrs[.size] as? NSNumber else {
        continue
    }
    // 16 kHz 16-bit mono: 32,000 bytes per second, 25 frames per second.
    let frames = Int(Double(size.intValue) / 32000.0 * 25.0)
    if frames < Defaults.minFrames {
        continue
    }
    paths.append(row.path)
    var start = 0
    while start < frames {
        let length = min(Defaults.windowFrames, frames - start)
        if Defaults.minFrames <= length {
            windows.append(Window(path: paths.count - 1, start: start, length: length))
        }
        start += Defaults.windowFrames
    }
}
if windows.isEmpty {
    print("error: no usable audio in \(manifestPath)")
    exit(1)
}
let itemPaths = paths
let itemWindows = windows

var teacher: MLXHawkNetwork? = nil
if teacherPath.isEmpty != true {
    guard let tw = try? HawkWeights.load(from: URL(fileURLWithPath: teacherPath)) else {
        print("error: cannot read teacher weights: \(teacherPath)")
        exit(1)
    }
    teacher = MLXHawkNetwork(weights: tw)
    print("teacher: \(teacherPath), residual stream after \(Defaults.teacherLayers) layers")
}

func teacherRows(_ feats: [[Float]], _ model: MLXHawkNetwork) -> [[Float]] {
    let x = MLXArray(feats.flatMap { $0 }, [1, feats.count, feats[0].count])
    let h = model.hidden(x, layerCount: Defaults.teacherLayers)
    eval(h)
    let dim = h.shape[2]
    let flat = h.asArray(Float.self)
    var rows: [[Float]] = []
    var t = 0
    while t < feats.count {
        rows.append(Array(flat[(t * dim)..<((t + 1) * dim)]))
        t += 1
    }
    return rows
}

if kmeansPath.isEmpty != true {
    var classes = Defaults.featureClusters
    if teacher != nil {
        classes = Defaults.teacherClusters
    }
    var frames: [[Float]] = []
    var order = Array(0..<paths.count).shuffled()
    while frames.count < Defaults.kmeansFrames && order.isEmpty != true {
        var feats = SpeechDataset.loadFeatures(path: paths[order.removeLast()]).features
        if feats.isEmpty {
            continue
        }
        feats = Array(feats.prefix(Defaults.windowFrames))
        var rows = feats
        if let model = teacher {
            rows = teacherRows(feats, model)
        }
        var k = 0
        while k < min(Defaults.kmeansFramesPerFile, rows.count) {
            frames.append(rows[Int.random(in: 0..<rows.count)])
            k += 1
        }
    }
    print("k-means: \(frames.count) frames (\(frames[0].count) dims) -> \(classes) classes, \(Defaults.kmeansIterations) iterations")
    let started = CFAbsoluteTimeGetCurrent()
    let book = ClusterCodebook.fit(frames: frames, k: classes, iterations: Defaults.kmeansIterations)
    let sample = Array(frames.prefix(50_000))
    let ids = ClusterCodebook.assign(features: MLXArray(sample.flatMap { $0 }, [sample.count, sample[0].count]), codebook: book.arrays())
    eval(ids)
    var usage = [Int](repeating: 0, count: book.count)
    for i in ids.asArray(Int32.self) {
        usage[Int(i)] += 1
    }
    let used = usage.filter { 0 < $0 }.count
    print("  \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - started)) s, classes used \(used)/\(book.count), largest class \(String(format: "%.1f", Double(usage.max() ?? 0) * 100.0 / Double(max(1, sample.count))))%")
    do {
        try book.save(to: URL(fileURLWithPath: kmeansPath))
        print("  ✓ codebook saved: \(kmeansPath)")
    } catch {
        print("  ✕ failed to save codebook: \(error)")
        exit(1)
    }
    exit(0)
}

guard let book = try? ClusterCodebook.load(from: URL(fileURLWithPath: targetsPath)) else {
    print("error: cannot read codebook: \(targetsPath)")
    exit(1)
}
let codebook = book.arrays()
var totalSeconds = 0.0
for w in windows {
    totalSeconds += Double(w.length) * 0.04
}
let network = MLXHawkNetwork(outputDim: book.count)
let trainer = MLXHawkTrainer(network: network, learningRate: Defaults.lrMax)
print("pretraining: \(book.count) classes, spans of \(Defaults.maskSpan) frames masked from start probability \(Defaults.maskStartProbability)")
print("  \(paths.count) utterances -> \(windows.count) windows of up to \(Defaults.windowFrames) frames (\(String(format: "%.1f", totalSeconds / 3600.0)) h/epoch), \(epochs) epochs, batch \(batchSize)")
print("  parameters \(network.exportWeights().parameterCount)")

let batchItems = batchSize
/// Shuffles windows within padded-length buckets, cuts batches, and shuffles the batch order.
func makeBatches() -> [[Int]] {
    var byBucket: [Int: [Int]] = [:]
    var i = 0
    while i < itemWindows.count {
        byBucket[((itemWindows[i].length + 31) / 32) * 32, default: []].append(i)
        i += 1
    }
    var batches: [[Int]] = []
    for (_, items) in byBucket {
        let shuffled = items.shuffled()
        var start = 0
        while start < shuffled.count {
            let end = min(start + batchItems, shuffled.count)
            batches.append(Array(shuffled[start..<end]))
            start = end
        }
    }
    return batches.shuffled()
}

final class BatchBuffer: @unchecked Sendable {
    var items: [[[Float]]]
    init(count: Int) {
        self.items = [[[Float]]](repeating: [], count: count)
    }
}

let loadWorkers = workers
@Sendable func loadBatch(_ indices: [Int]) -> [[[Float]]] {
    let buffer = BatchBuffer(count: indices.count)
    let workerCount = max(1, min(loadWorkers, indices.count))
    DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
        var k = worker
        while k < indices.count {
            let w = itemWindows[indices[k]]
            buffer.items[k] = autoreleasepool {
                let feats = SpeechDataset.loadFeatures(path: itemPaths[w.path]).features
                if feats.count <= w.start {
                    return []
                }
                return Array(feats[w.start..<min(feats.count, w.start + w.length)])
            }
            k += workerCount
        }
    }
    return buffer.items.filter { Defaults.minFrames <= $0.count }
}

final class PrefetchBox: @unchecked Sendable {
    var value: [[[Float]]] = []
}

let batchesPerEpoch = makeBatches().count
let totalSteps = max(1, epochs * batchesPerEpoch)
let warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin)
print("  steps: \(totalSteps) (warmup \(warmupSteps)), learning rate \(Defaults.lrMax) -> \(Defaults.lrMin)")

var globalStep = 0
let startAll = CFAbsoluteTimeGetCurrent()
var ep = 1
while ep <= epochs {
    let batches = makeBatches()
    let epStart = CFAbsoluteTimeGetCurrent()
    var lossSum: Float = 0.0
    var windowLoss: Float = 0.0
    var windowCount = 0
    var waitSeconds = 0.0
    var current = loadBatch(batches[0])
    var b = 0
    while b < batches.count {
        let box = PrefetchBox()
        let group = DispatchGroup()
        if (b + 1) < batches.count {
            let next = batches[b + 1]
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                box.value = loadBatch(next)
                group.leave()
            }
        }
        globalStep += 1
        trainer.updateLearningRate(scheduler.learningRate(step: globalStep, totalSteps: totalSteps, warmupSteps: warmupSteps))
        let loss = trainer.trainBatchMaskedCluster(
            featuresBatch: current, codebook: codebook,
            maskStartProbability: Defaults.maskStartProbability, maskSpan: Defaults.maskSpan,
            teacher: teacher, teacherLayers: Defaults.teacherLayers)
        if loss.isFinite != true {
            print("  ✕ loss is not finite (step \(globalStep)); stopping")
            exit(1)
        }
        lossSum += loss
        windowLoss += loss
        windowCount += 1
        if globalStep % Defaults.logEvery == 0 {
            print("    step \(globalStep): loss \(String(format: "%.4f", windowLoss / Float(max(1, windowCount)))), batch \(b + 1)/\(batches.count), \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - epStart)) s (prefetch wait \(String(format: "%.0f", waitSeconds)) s)")
            windowLoss = 0.0
            windowCount = 0
        }
        let waitStart = CFAbsoluteTimeGetCurrent()
        group.wait()
        waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
        current = box.value
        b += 1
    }
    print("  Epoch [\(ep)/\(epochs)] - loss: \(String(format: "%.4f", lossSum / Float(max(1, batches.count)))) (time: \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - epStart)) s, prefetch wait \(String(format: "%.0f", waitSeconds)) s)")
    var path = exportPath
    if ep < epochs {
        path = exportPath + ".ep\(ep).json"
    }
    do {
        try network.exportWeights().save(to: URL(fileURLWithPath: path))
        print("    ✓ weights saved: \(path)")
    } catch {
        print("    ✕ failed to save weights: \(error)")
    }
    ep += 1
}
print("pretraining done (\(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - startAll)) s)")
