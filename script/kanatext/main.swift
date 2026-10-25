import Foundation
import Haitaka
import MLX
import MLXNN
import MLXOptimizers

setbuf(stdout, nil)

// Streaming kana-to-text converter: pronunciation kana (the acoustic model's targets) → text with kanji, Latin letters
// and punctuation, unit by unit (a unit ends at a punctuation mark).
//   kanatext prepare --english-dict <cmudict> <in.tsv> <out.tsv>      stream \t text  →  stream \t kana \t text
//   kanatext train --out <dir> [-e epochs] [--eval <prepared.tsv>]... <prepared train.tsv>
//   kanatext eval --model <dir> [--kana-from <eval dump.jsonl>] <prepared.tsv>...
// Training reads consecutive lines of a stream as one sequence (up to `chunkKana` kana), so the recurrent state learns
// to carry context across units. CER compares NFKC text without punctuation and spaces (`scoringForm`).

enum Defaults {
    static let epochs = 30
    static let lrMax: Float = 0.0005
    static let warmupSteps = 1000
    static let batchTokens = 12288
    static let chunkKana = 384
    static let maxLineKana = 1024
    static let evalBatch = 64
    static let cacheLimitBytes = 2 << 30
    static let smoothing: Float = 0.1
    static let clipNorm: Float = 1.0
    static let seed: UInt64 = 1
    static let examples = 5
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func readLines(_ path: String) -> [String] {
    guard let data = FileManager.default.contents(atPath: path), let text = String(data: data, encoding: .utf8) else {
        fail("cannot read \(path)")
    }
    return text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
}

struct Pair {
    let label: String
    let kana: String
    let text: String
}

func readPairs(_ path: String) -> [Pair] {
    return readLines(path).compactMap { line in
        let f = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        if f.count < 3 {
            return nil
        }
        return Pair(label: f[0], kana: f[1], text: f[2])
    }
}

let scoringDropped: Set<Character> = ["、", "。", "？", "！", "，", "．", "・", "「", "」", "『", "』", " ", "　"]

func scoringForm(_ text: String) -> [Character] {
    let folded = text.precomposedStringWithCompatibilityMapping
    return folded.filter { c in
        if scoringDropped.contains(c) {
            return false
        }
        return c.isPunctuation != true && c.isWhitespace != true
    }
}

func editDistance(_ a: [Character], _ b: [Character]) -> Int {
    if a.isEmpty {
        return b.count
    }
    if b.isEmpty {
        return a.count
    }
    var prev = Array(0...b.count)
    var cur = [Int](repeating: 0, count: b.count + 1)
    var i = 1
    while i <= a.count {
        cur[0] = i
        var j = 1
        while j <= b.count {
            var best = min(prev[j] + 1, cur[j - 1] + 1)
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            best = min(best, prev[j - 1] + cost)
            cur[j] = best
            j += 1
        }
        swap(&prev, &cur)
        i += 1
    }
    return prev[b.count]
}

struct Chunk {
    var source: [Int] = []
    var sourceUnit: [Int] = []
    var previous: [Int] = []
    var target: [Int] = []
    var targetUnit: [Int] = []
    var units = 0
}

/// Consecutive lines of each stream, split into units, packed into chunks of about `chunkKana` kana.
func makeChunks(_ pairs: [Pair], _ vocabulary: KanaTextVocabulary) -> [Chunk] {
    var chunks: [Chunk] = []
    var current = Chunk()
    var stream = ""
    func close() {
        if current.units != 0 {
            chunks.append(current)
        }
        current = Chunk()
    }
    for p in pairs {
        if p.kana.count > Defaults.maxLineKana {
            continue
        }
        if p.label != stream || Defaults.chunkKana < current.source.count + p.kana.count {
            close()
            stream = p.label
        }
        for u in KanaTextUnits.paired(kana: p.kana, text: p.text) {
            let k = vocabulary.encodeInput(u.kana)
            let t = vocabulary.encodeOutput(u.text) + [KanaTextVocabulary.eos]
            current.source += k
            current.sourceUnit += [Int](repeating: current.units, count: k.count)
            current.previous += [current.units == 0 ? KanaTextVocabulary.bos : KanaTextVocabulary.eos] + t.dropLast()
            current.target += t
            current.targetUnit += [Int](repeating: current.units, count: t.count)
            current.units += 1
        }
    }
    close()
    return chunks
}

func roundUp(_ n: Int) -> Int {
    return ((max(1, n) + 31) / 32) * 32
}

/// Chunks of similar length, at most `budget` padded positions per batch.
func makeBatches(_ chunks: [Chunk], budget: Int) -> [[Int]] {
    let order = chunks.indices.sorted { a, b in
        return max(chunks[a].source.count, chunks[a].target.count) < max(chunks[b].source.count, chunks[b].target.count)
    }
    var batches: [[Int]] = []
    var current: [Int] = []
    var longest = 0
    for i in order {
        let len = roundUp(max(chunks[i].source.count, chunks[i].target.count))
        let wider = max(longest, len)
        if current.isEmpty != true && budget < wider * (current.count + 1) {
            batches.append(current)
            current = []
            longest = 0
        }
        current.append(i)
        longest = max(longest, len)
    }
    if current.isEmpty != true {
        batches.append(current)
    }
    return batches
}

func padded(_ rows: [[Int]], width: Int, fill: Int) -> MLXArray {
    var flat = [Int32](repeating: Int32(fill), count: rows.count * width)
    var r = 0
    while r < rows.count {
        var c = 0
        while c < rows[r].count {
            flat[r * width + c] = Int32(rows[r][c])
            c += 1
        }
        r += 1
    }
    return MLXArray(flat, [rows.count, width])
}

/// Additive mask [B, 1, T, S]: each target position sees the source positions of its own unit.
func unitMask(_ batch: [Chunk], targetWidth: Int, sourceWidth: Int) -> MLXArray {
    var values = [Float](repeating: -1e9, count: batch.count * targetWidth * sourceWidth)
    var b = 0
    while b < batch.count {
        let c = batch[b]
        var t = 0
        while t < c.targetUnit.count {
            var s = 0
            while s < c.sourceUnit.count {
                if c.sourceUnit[s] == c.targetUnit[t] {
                    values[(b * targetWidth + t) * sourceWidth + s] = 0.0
                }
                s += 1
            }
            t += 1
        }
        b += 1
    }
    return MLXArray(values, [batch.count, 1, targetWidth, sourceWidth])
}

func saveModel(_ model: MLXKanaTextModel, vocabulary: KanaTextVocabulary, to dir: String) throws {
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    var arrays: [String: MLXArray] = [:]
    for (k, v) in model.parameters().flattened() {
        arrays[k] = v
    }
    try MLX.save(arrays: arrays, url: URL(fileURLWithPath: dir + "/weights.safetensors"))
    try JSONEncoder().encode(vocabulary).write(to: URL(fileURLWithPath: dir + "/vocabulary.json"))
    try JSONEncoder().encode(model.shape).write(to: URL(fileURLWithPath: dir + "/shape.json"))
}

func loadModel(from dir: String) throws -> (MLXKanaTextModel, KanaTextVocabulary) {
    let vocabulary = try JSONDecoder().decode(KanaTextVocabulary.self, from: Data(contentsOf: URL(fileURLWithPath: dir + "/vocabulary.json")))
    let shape = try JSONDecoder().decode(KanaTextShape.self, from: Data(contentsOf: URL(fileURLWithPath: dir + "/shape.json")))
    let model = MLXKanaTextModel(shape: shape, inputVocabulary: vocabulary.input.count, outputVocabulary: vocabulary.output.count)
    let arrays = try loadArrays(url: URL(fileURLWithPath: dir + "/weights.safetensors"))
    model.update(parameters: ModuleParameters.unflattened(arrays))
    eval(model)
    return (model, vocabulary)
}

/// Converts each line on its own (state reset per line); returns (CER %, outputs).
func evaluate(_ model: MLXKanaTextModel, _ vocabulary: KanaTextVocabulary, _ pairs: [Pair], kana: [String]) -> (Double, [String]) {
    let order = pairs.indices.sorted { kana[$0].count < kana[$1].count }
    var outputs = [String](repeating: "", count: pairs.count)
    var start = 0
    while start < order.count {
        let rows = Array(order[start..<min(order.count, start + Defaults.evalBatch)])
        var sources: [[Int]] = []
        var spans: [[Range<Int>]] = []
        for r in rows {
            var ids: [Int] = []
            var units: [Range<Int>] = []
            for u in KanaTextUnits.splitKana(kana[r]) {
                let k = vocabulary.encodeInput(u)
                units.append(ids.count..<(ids.count + k.count))
                ids += k
            }
            if ids.isEmpty {
                ids = [KanaTextVocabulary.unknown]
                units = [0..<1]
            }
            sources.append(ids)
            spans.append(units)
        }
        let width = roundUp(sources.map { $0.count }.max() ?? 1)
        let ids = model.greedy(padded(sources, width: width, fill: KanaTextVocabulary.pad), units: spans)
        var i = 0
        while i < rows.count {
            outputs[rows[i]] = ids[i].map { vocabulary.decodeOutput($0) }.joined()
            i += 1
        }
        start += Defaults.evalBatch
    }
    var errors = 0
    var total = 0
    var i = 0
    while i < pairs.count {
        let ref = scoringForm(pairs[i].text)
        errors += editDistance(scoringForm(outputs[i]), ref)
        total += ref.count
        i += 1
    }
    return (100.0 * Double(errors) / Double(max(1, total)), outputs)
}

func printExamples(_ pairs: [Pair], _ outputs: [String], count: Int) {
    var i = 0
    let stride = max(1, pairs.count / max(1, count))
    while i < pairs.count && i / stride < count {
        print("    ref: \(pairs[i].text)")
        print("    hyp: \(outputs[i])")
        i += stride
    }
}

// MARK: - commands

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    fail("usage: kanatext prepare|train|eval ...")
}
var options: [String: [String]] = [:]
var positional: [String] = []
var argIdx = 1
while argIdx < args.count {
    let a = args[argIdx]
    if a.hasPrefix("-") && argIdx + 1 < args.count {
        options[a, default: []].append(args[argIdx + 1])
        argIdx += 2
        continue
    }
    positional.append(a)
    argIdx += 1
}

switch command {
case "prepare":
    guard let dictPath = options["--english-dict"]?.first, positional.count == 2 else {
        fail("usage: kanatext prepare --english-dict <cmudict> <in.tsv> <out.tsv>")
    }
    guard let english = try? EnglishPronunciations(contentsOfFile: dictPath) else {
        fail("cannot read \(dictPath)")
    }
    let converter = KanjiConverter(english: english)
    let lines = readLines(positional[0])
    var out = [String](repeating: "", count: lines.count)
    DispatchQueue.concurrentPerform(iterations: 16) { worker in
        var i = worker
        while i < lines.count {
            let f = lines[i].split(separator: "\t", maxSplits: 1).map(String.init)
            if f.count == 2 {
                let kana = converter.convertToHiragana(f[1])
                if kana.isEmpty != true {
                    out[i] = "\(f[0])\t\(kana)\t\(f[1])"
                }
            }
            i += 16
        }
    }
    let kept = out.filter { $0.isEmpty != true }
    try kept.joined(separator: "\n").appending("\n").write(toFile: positional[1], atomically: true, encoding: .utf8)
    print("\(kept.count) of \(lines.count) lines → \(positional[1])")

case "train":
    guard let outDir = options["--out"]?.first, positional.count == 1 else {
        fail("usage: kanatext train --out <dir> [-e epochs] [--eval <prepared.tsv>]... <prepared train.tsv>")
    }
    Device.setDefault(device: .gpu)
    MLX.GPU.set(cacheLimit: Defaults.cacheLimitBytes)
    MLXRandom.seed(Defaults.seed)
    let epochs = Int(options["-e"]?.first ?? "") ?? Defaults.epochs
    let pairs = readPairs(positional[0])
    let evalSets = (options["--eval"] ?? []).map { (name: ($0 as NSString).lastPathComponent, pairs: readPairs($0)) }
    let vocabulary = KanaTextVocabulary(pairs: pairs.map { (kana: $0.kana, text: $0.text) })
    let shape = KanaTextShape()
    let model = MLXKanaTextModel(shape: shape, inputVocabulary: vocabulary.input.count, outputVocabulary: vocabulary.output.count)
    eval(model)
    let parameterCount = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
    let chunks = makeChunks(pairs, vocabulary)
    let unitCount = chunks.reduce(0) { $0 + $1.units }
    print("lines \(pairs.count), chunks \(chunks.count), units \(unitCount), input vocabulary \(vocabulary.input.count), output vocabulary \(vocabulary.output.count)")
    print("model \(shape.modelDim) dim, \(shape.encoderLayers)+\(shape.decoderLayers) layers, \(parameterCount) parameters")
    var batches = makeBatches(chunks, budget: Defaults.batchTokens)
    let totalSteps = epochs * batches.count
    print("batches \(batches.count) per epoch, \(totalSteps) steps")
    let optimizer = ArrayLRAdam(learningRate: Defaults.lrMax, betas: (0.9, 0.98), eps: 1e-9)
    let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: 0.0)
    let lossAndGrad = valueAndGrad(model: model) { (m: MLXKanaTextModel, a: [MLXArray]) -> [MLXArray] in
        [MLXKanaTextModel.loss(logits: m.decode(a[1], memory: m.encode(a[0]), mask: a[3]), targets: a[2], smoothing: Defaults.smoothing)]
    }
    var order = SplitMix64(seed: Defaults.seed)
    var step = 0
    var best = Double.greatestFiniteMagnitude
    var ep = 1
    while ep <= epochs {
        let start = CFAbsoluteTimeGetCurrent()
        batches.shuffle(using: &order)
        var lossSum: Float = 0.0
        for batch in batches {
            step += 1
            optimizer.learningRate = MLXArray(scheduler.learningRate(step: step, totalSteps: totalSteps, warmupSteps: Defaults.warmupSteps))
            let members = batch.map { chunks[$0] }
            let sw = roundUp(members.map { $0.source.count }.max() ?? 1)
            let tw = roundUp(members.map { $0.target.count }.max() ?? 1)
            let source = padded(members.map { $0.source }, width: sw, fill: KanaTextVocabulary.pad)
            let previous = padded(members.map { $0.previous }, width: tw, fill: KanaTextVocabulary.pad)
            let targets = padded(members.map { $0.target }, width: tw, fill: KanaTextVocabulary.pad)
            let mask = unitMask(members, targetWidth: tw, sourceWidth: sw)
            let (loss, grads) = lossAndGrad(model, [source, previous, targets, mask])
            let (clipped, _) = clipGradNorm(gradients: grads, maxNorm: Defaults.clipNorm)
            optimizer.update(model: model, gradients: clipped)
            eval(model, optimizer, loss)
            lossSum += loss[0].item(Float.self)
        }
        print(String(format: "epoch %d/%d loss %.4f (%.0f s, peak %d MB)", ep, epochs, lossSum / Float(batches.count),
                     CFAbsoluteTimeGetCurrent() - start, MLX.GPU.peakMemory >> 20))
        var selection = 0.0
        for set in evalSets {
            let (cer, outputs) = evaluate(model, vocabulary, set.pairs, kana: set.pairs.map { $0.kana })
            print(String(format: "  %@: CER %.2f%% (%d)", set.name, cer, set.pairs.count))
            if set.name.contains("dev") && selection == 0.0 {
                selection = cer
            }
            if ep == epochs {
                printExamples(set.pairs, outputs, count: Defaults.examples)
            }
        }
        try saveModel(model, vocabulary: vocabulary, to: outDir + "/ep\(ep)")
        if 0.0 < selection && selection < best {
            best = selection
            try saveModel(model, vocabulary: vocabulary, to: outDir + "/best")
            print("  best so far (dev CER \(String(format: "%.2f", selection))%)")
        }
        ep += 1
    }
    try saveModel(model, vocabulary: vocabulary, to: outDir + "/last")

case "eval":
    guard let modelDir = options["--model"]?.first, positional.isEmpty != true else {
        fail("usage: kanatext eval --model <dir> [--kana-from <eval dump.jsonl>] <prepared.tsv>...")
    }
    Device.setDefault(device: .gpu)
    MLX.GPU.set(cacheLimit: Defaults.cacheLimitBytes)
    let (model, vocabulary) = try loadModel(from: modelDir)
    var predicted: [String: String] = [:]
    for dump in options["--kana-from"] ?? [] {
        for line in readLines(dump) {
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let id = obj["fileId"] as? String, let kana = obj["predKana"] as? String else {
                continue
            }
            predicted[id] = kana
        }
    }
    for path in positional {
        let pairs = readPairs(path)
        let (cer, outputs) = evaluate(model, vocabulary, pairs, kana: pairs.map { $0.kana })
        var line = String(format: "%@: reference kana CER %.2f%%", (path as NSString).lastPathComponent, cer)
        let covered = pairs.filter { predicted[$0.label] != nil }
        if covered.isEmpty != true {
            let (e2e, _) = evaluate(model, vocabulary, covered, kana: covered.map { predicted[$0.label]! })
            line += String(format: " / acoustic kana CER %.2f%% (%d)", e2e, covered.count)
        }
        print(line + " (\(pairs.count))")
        printExamples(pairs, outputs, count: Defaults.examples)
    }

default:
    fail("unknown command \(command)")
}
