import Foundation
import Haitaka
import MLX
import MLXNN
import MLXOptimizers

// kanatext listen: the decoder that listens. The acoustic model (frozen, outside the trained module) runs on each
// batch; its CTC kana feeds the kana encoder and its normalized output frames feed the `listen` cross-attention.
// The decoder state starts fresh per utterance.
//   kanatext listen --acoustic <weights.safetensors> --init <kanatext dir> --mode kana|audio|both --out <dir>
//                   [-e epochs] [--english-dict <cmudict>] [--kana-lm <kana text>] [--noise-bank <jsonl>]
//                   [--eval <manifest.jsonl>]... [<train manifest.jsonl>]
// kana: the kanatext layers on recognized kana; audio: no kana, the decoder attends to the frames; both: kana and
// frames, the frame attention starting as the identity. -e 0 only evaluates the initial model.
// With kana, the utterance is split into units at punctuation as in kanatext: each unit's text ends with eos and
// attends to its own kana, while the frame attention sees the whole utterance. Training pairs the greedy CTC kana's
// units with the text's when their counts match, and otherwise the reference kana's (from --english-dict, which
// training with kana needs). Evaluation reads kana from the CTC beam fused with the kana n-gram LM of --kana-lm
// (greedy without it); an utterance whose kana is only punctuation gets empty text. audio: one utterance is one unit.
// Writes fileId, reference, output and CTC kana per utterance to <dir>/ep<N>/outputs_<eval name>.tsv.

enum ListenDefaults {
    static let epochs = 4
    static let lrMax: Float = 0.0005
    static let warmupSteps = 1000
    static let frameBudget = 32768
    static let maxBatch = 256
    static let workers = 8
    static let cacheLimitBytes = 8 << 30
    static let maxCharsPerFrame = 0.5
    static let logEvery = 500
}

struct ListenSample {
    let path: String
    let fileId: String
    let text: String
    let frames: Int
}

nonisolated(unsafe) let listenLaugh = try! NSRegularExpression(pattern: "[(（][^()（）]{0,6}?笑[^()（）]{0,6}?[)）]")

/// The text the decoder learns to write: laugh annotations and ♪ dropped, runs of spaces collapsed.
func listenText(_ raw: String) -> String {
    let range = NSRange(raw.startIndex..., in: raw)
    let dropped = listenLaugh.stringByReplacingMatches(in: raw, range: range, withTemplate: "").replacingOccurrences(of: "♪", with: "")
    return dropped.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
}

func readListenSamples(_ path: String) -> [ListenSample] {
    var pairs: [(path: String, text: String)] = []
    for line in readLines(path) {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let p = obj["path"] as? String, let t = obj["text"] as? String else {
            continue
        }
        pairs.append((path: p, text: t))
    }
    let dataset = SpeechDataset.lazyFromManifest(pairs: pairs, workers: ListenDefaults.workers)
    return dataset.metaSamples.map { m in
        ListenSample(path: m.path, fileId: ((m.path as NSString).lastPathComponent as NSString).deletingPathExtension,
                     text: listenText(m.rawText), frames: m.frameCount)
    }
}

/// Utterances of similar length, at most `frameBudget` padded frames and `maxBatch` utterances per batch.
func listenBatches(_ samples: [ListenSample]) -> [[Int]] {
    let order = samples.indices.sorted { samples[$0].frames < samples[$1].frames }
    var batches: [[Int]] = []
    var current: [Int] = []
    for i in order {
        let width = roundUp(samples[i].frames)
        if current.isEmpty != true && (ListenDefaults.maxBatch <= current.count || ListenDefaults.frameBudget < width * (current.count + 1)) {
            batches.append(current)
            current = []
        }
        current.append(i)
    }
    if current.isEmpty != true {
        batches.append(current)
    }
    return batches
}

final class ListenFeatures: @unchecked Sendable {
    var items: [[[Float]]]
    init(count: Int) {
        items = [[[Float]]](repeating: [], count: count)
    }
}

func listenFeatures(_ samples: [ListenSample], _ indices: [Int], augmenter: FeatureAugmenter?) -> [[[Float]]] {
    let buffer = ListenFeatures(count: indices.count)
    let workers = max(1, min(ListenDefaults.workers, indices.count))
    DispatchQueue.concurrentPerform(iterations: workers) { worker in
        var i = worker
        while i < indices.count {
            let path = samples[indices[i]].path
            buffer.items[i] = autoreleasepool {
                SpeechDataset.loadFeatures(path: path, pcmTransform: augmenter?.pcmTransform).features
            }
            i += workers
        }
    }
    return buffer.items
}

/// Features [B, F, D] zero-padded to a multiple of 32 frames, with each row's frame count.
func listenPadded(_ batch: [[[Float]]]) -> (MLXArray, [Int]) {
    let counts = batch.map { $0.count }
    let width = roundUp(counts.max() ?? 1)
    let dim = StreamingFeatureFrontEnd.inputDim
    var flat = [Float](repeating: 0.0, count: batch.count * width * dim)
    var b = 0
    while b < batch.count {
        var t = 0
        while t < batch[b].count {
            let offset = (b * width + t) * dim
            var d = 0
            while d < dim {
                flat[offset + d] = batch[b][t][d]
                d += 1
            }
            t += 1
        }
        b += 1
    }
    return (MLXArray(flat, [batch.count, width, dim]), counts)
}

/// The acoustic model's normalized output frames, its CTC logits and the greedy CTC kana of each row.
func listenAcoustic(_ acoustic: MLXHawkNetwork, _ kanaVocabulary: TextVocabulary, features: MLXArray, counts: [Int]) -> (MLXArray, MLXArray, [String]) {
    let (output, logits) = acoustic.encoded(features, frameCounts: MLXArray(counts.map { Int32($0) }))
    let best = argMax(logits, axis: -1)
    eval(output, logits, best)
    let width = features.dim(1)
    let ids = best.asArray(Int32.self)
    var kana: [String] = []
    var r = 0
    while r < counts.count {
        var s = ""
        var previous: Int32 = -1
        var t = 0
        while t < counts[r] {
            let id = ids[r * width + t]
            if id != previous && id != Int32(TextVocabulary.padId) {
                s.append(kanaVocabulary.char(for: Int(id)))
            }
            previous = id
            t += 1
        }
        kana.append(s)
        r += 1
    }
    return (output, logits, kana)
}

final class ListenKana: @unchecked Sendable {
    var items: [String]
    init(count: Int) {
        items = [String](repeating: "", count: count)
    }
}

/// CTC beam search fused with the kana n-gram LM over each row's first `counts[r]` frames of `logits` [B, F, V].
func listenBeamKana(_ logits: MLXArray, counts: [Int], _ kanaVocabulary: TextVocabulary, _ lm: KanaNgramLM) -> [String] {
    let logp = logSoftmax(logits, axis: -1)
    eval(logp)
    let width = logits.dim(1)
    let v = logits.dim(2)
    let flat = logp.asArray(Float.self)
    let buffer = ListenKana(count: counts.count)
    let workers = max(1, min(ListenDefaults.workers, counts.count))
    DispatchQueue.concurrentPerform(iterations: workers) { worker in
        let decoder = CTCDecoder(vocabulary: kanaVocabulary, languageModel: lm)
        var r = worker
        while r < counts.count {
            let frames = (0..<counts[r]).map { t in Array(flat[((r * width + t) * v)..<((r * width + t + 1) * v)]) }
            buffer.items[r] = decoder.decode(logProbs: frames).text
            r += workers
        }
    }
    return buffer.items
}

/// One utterance as a kanatext chunk. The recognized kana's units are paired with the text's when their counts
/// match; otherwise the reference kana's are (`KanaTextUnits.paired` keeps the line whole if those differ too).
/// Returns the chunk and whether it reads the recognized kana.
func listenChunk(recognized: String, reference: String, text: String, _ vocabulary: KanaTextVocabulary) -> (Chunk, Bool) {
    var kana = reference
    let units = KanaTextUnits.splitKana(recognized)
    let recognizedFits = scoringForm(recognized).isEmpty != true && units.count == KanaTextUnits.splitText(text).count
    if recognizedFits {
        kana = recognized
    }
    var chunk = Chunk()
    for u in KanaTextUnits.paired(kana: kana, text: text) {
        chunk.append(kana: u.kana, text: u.text, vocabulary)
    }
    return (chunk, recognizedFits)
}

/// Greedy decoding of every sample. With kana, each utterance is decoded unit by unit (`MLXKanaTextModel.greedy`);
/// without, as one unit (`greedyUtterances`). Utterances whose CTC kana is only punctuation get empty text. Returns
/// the character CER (%), units cut by the repetition stop and empty outputs.
func listenEvaluate(_ model: MLXKanaTextModel, _ vocabulary: KanaTextVocabulary, _ acoustic: MLXHawkNetwork,
                    _ kanaVocabulary: TextVocabulary, _ lm: KanaNgramLM?, _ samples: [ListenSample],
                    outputPath: String) throws -> (Double, Int, Int) {
    var rows: [String] = []
    var errors = 0
    var total = 0
    var cut = 0
    var empty = 0
    for batch in listenBatches(samples) {
        let (features, counts) = listenPadded(listenFeatures(samples, batch, augmenter: nil))
        let (audio, logits, greedyKana) = listenAcoustic(acoustic, kanaVocabulary, features: features, counts: counts)
        var kana = greedyKana
        if let lm {
            kana = listenBeamKana(logits, counts: counts, kanaVocabulary, lm)
        }
        let silent = kana.map { scoringForm($0).isEmpty }
        var listened: MLXArray? = nil
        if 0 < model.shape.audioDim {
            listened = audio
        }
        var texts: [String] = []
        if model.shape.kanaInput {
            var sources: [[Int]] = []
            var spans: [[Range<Int>]] = []
            for k in kana {
                var ids: [Int] = []
                var units: [Range<Int>] = []
                for u in KanaTextUnits.splitKana(k) {
                    let encoded = vocabulary.encodeInput(u)
                    units.append(ids.count..<(ids.count + encoded.count))
                    ids += encoded
                }
                if ids.isEmpty {
                    ids = [KanaTextVocabulary.unknown]
                    units = [0..<1]
                }
                sources.append(ids)
                spans.append(units)
            }
            let width = roundUp(sources.map { $0.count }.max() ?? 1)
            let result = model.greedy(padded(sources, width: width, fill: KanaTextVocabulary.pad), units: spans,
                                      audio: listened, frameCounts: counts)
            cut += result.cut
            texts = result.units.map { u in u.map { vocabulary.decodeOutput($0) }.joined() }
        } else {
            let limits = kana.indices.map { r in
                if silent[r] {
                    return 0
                }
                return 2 * kana[r].count + 8
            }
            let allowed = kana.map { max(2, KanaTextRepetition.longest(vocabulary.encodeInput($0))) }
            let result = model.greedyUtterances(source: nil, sourceCounts: [], audio: listened, frameCounts: counts,
                                                limits: limits, allowed: allowed)
            cut += result.cut
            texts = result.texts.map { vocabulary.decodeOutput($0) }
        }
        var i = 0
        while i < batch.count {
            let s = samples[batch[i]]
            var text = texts[i]
            if silent[i] {
                text = ""
                empty += 1
            }
            let ref = scoringForm(s.text)
            errors += editDistance(scoringForm(text), ref)
            total += ref.count
            rows.append("\(s.fileId)\t\(s.text)\t\(text)\t\(kana[i])")
            i += 1
        }
    }
    try rows.joined(separator: "\n").appending("\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
    return (100.0 * Double(errors) / Double(max(1, total)), cut, empty)
}

func runListen(options: [String: [String]], positional: [String]) throws {
    guard let acousticPath = options["--acoustic"]?.first, let initDir = options["--init"]?.first,
          let mode = options["--mode"]?.first, ["kana", "audio", "both"].contains(mode), let outDir = options["--out"]?.first else {
        fail("usage: kanatext listen --acoustic <weights> --init <kanatext dir> --mode kana|audio|both --out <dir> [-e epochs] [--english-dict <cmudict>] [--kana-lm <kana text>] [--noise-bank <jsonl>] [--eval <manifest>]... [<train manifest>]")
    }
    let epochs = Int(options["-e"]?.first ?? "") ?? ListenDefaults.epochs
    if 0 < epochs && positional.count != 1 {
        fail("kanatext listen: training needs one train manifest")
    }
    Device.setDefault(device: .gpu)
    MLX.GPU.set(cacheLimit: ListenDefaults.cacheLimitBytes)
    MLXRandom.seed(Defaults.seed)

    let weights = try HawkWeights.load(from: URL(fileURLWithPath: acousticPath))
    guard let kanaVocabulary = weights.vocabulary else {
        fail("kanatext listen: \(acousticPath) has no kana readout")
    }
    let acoustic = MLXHawkNetwork(weights: weights)
    eval(acoustic)

    let vocabulary = try JSONDecoder().decode(KanaTextVocabulary.self, from: Data(contentsOf: URL(fileURLWithPath: initDir + "/vocabulary.json")))
    var shape = try JSONDecoder().decode(KanaTextShape.self, from: Data(contentsOf: URL(fileURLWithPath: initDir + "/shape.json")))
    shape.kanaInput = mode != "audio"
    shape.audioDim = 0
    if mode != "kana" {
        shape.audioDim = acoustic.modelDim
    }
    let model = MLXKanaTextModel(shape: shape, inputVocabulary: vocabulary.input.count, outputVocabulary: vocabulary.output.count)
    let own = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    let saved = try loadArrays(url: URL(fileURLWithPath: initDir + "/weights.safetensors"))
    let kept = saved.filter { own[$0.key]?.shape == $0.value.shape }
    model.update(parameters: ModuleParameters.unflattened(kept))
    eval(model)
    let parameterCount = own.values.reduce(0) { $0 + $1.size }
    print("mode \(mode): \(parameterCount) parameters, \(kept.count) of \(own.count) arrays from \(initDir); acoustic \(acousticPath) (\(acoustic.modelDim) wide, frozen)")

    var lm: KanaNgramLM? = nil
    if let lmPath = options["--kana-lm"]?.first {
        guard let loaded = KanaNgramLM(contentsOfFile: lmPath, vocabulary: kanaVocabulary) else {
            fail("kanatext listen: cannot read \(lmPath)")
        }
        lm = loaded
    }
    let evalSets = (options["--eval"] ?? []).map { (name: (($0 as NSString).lastPathComponent as NSString).deletingPathExtension, samples: readListenSamples($0)) }
    func evaluateAll(_ dir: String) throws -> Double {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        var selection = 0.0
        for set in evalSets {
            let (cer, cut, empty) = try listenEvaluate(model, vocabulary, acoustic, kanaVocabulary, lm, set.samples,
                                                       outputPath: dir + "/outputs_\(set.name).tsv")
            print(String(format: "  %@: CER %.2f%% (%d utterances, %d cut, %d without CTC kana)", set.name, cer, set.samples.count, cut, empty))
            if selection == 0.0 {
                selection = cer
            }
        }
        return selection
    }
    if epochs == 0 {
        _ = try evaluateAll(outDir + "/ep0")
        try saveModel(model, vocabulary: vocabulary, to: outDir + "/ep0")
        return
    }

    var noiseBank: NoiseBank? = nil
    if let bankPath = options["--noise-bank"]?.first {
        noiseBank = try NoiseBank(manifestPath: bankPath)
    }
    let augmenter = FeatureAugmenter(speed: false, specAugment: false, noiseBank: noiseBank)
    let loaded = readListenSamples(positional[0])
    let samples = loaded.filter { s in
        s.text.isEmpty != true && Double(s.text.count) <= ListenDefaults.maxCharsPerFrame * Double(s.frames)
    }
    var batches = listenBatches(samples)
    let totalSteps = epochs * batches.count
    print("train \(samples.count) utterances (\(loaded.count - samples.count) dropped: empty or more than \(ListenDefaults.maxCharsPerFrame) characters per frame), \(batches.count) batches per epoch, \(totalSteps) steps")
    var referenceKana: [String] = []
    if shape.kanaInput {
        guard let dictPath = options["--english-dict"]?.first, let english = try? EnglishPronunciations(contentsOfFile: dictPath) else {
            fail("kanatext listen: training with kana needs --english-dict <cmudict>")
        }
        let converter = KanjiConverter(english: english)
        let buffer = ListenKana(count: samples.count)
        DispatchQueue.concurrentPerform(iterations: 16) { worker in
            var i = worker
            while i < samples.count {
                buffer.items[i] = converter.convertToHiragana(samples[i].text)
                i += 16
            }
        }
        referenceKana = buffer.items
    }

    let optimizer = ArrayLRAdam(learningRate: ListenDefaults.lrMax, betas: (0.9, 0.98), eps: 1e-9)
    let scheduler = CosineLRScheduler(lrMax: ListenDefaults.lrMax, lrMin: 0.0)
    let lossAndGrad = valueAndGrad(model: model) { (m: MLXKanaTextModel, a: [MLXArray]) -> [MLXArray] in
        var memory: MLXArray? = nil
        var mask: MLXArray? = nil
        if m.shape.kanaInput {
            memory = m.encode(a[0])
            mask = a[3]
        }
        var audio: MLXArray? = nil
        var audioMask: MLXArray? = nil
        if 0 < m.shape.audioDim {
            audio = a[4]
            audioMask = a[5]
        }
        let logits = m.decode(a[1], memory: memory, mask: mask, audio: audio, audioMask: audioMask)
        return [MLXKanaTextModel.loss(logits: logits, targets: a[2], smoothing: Defaults.smoothing)]
    }
    var order = SplitMix64(seed: Defaults.seed)
    var step = 0
    var best = Double.greatestFiniteMagnitude
    var ep = 1
    while ep <= epochs {
        let start = CFAbsoluteTimeGetCurrent()
        augmenter.epochSalt = UInt64(ep)
        batches.shuffle(using: &order)
        var lossSum: Float = 0.0
        var windowSum: Float = 0.0
        var recognizedUtterances = 0
        let prefetch = ListenFeatures(count: 0)
        prefetch.items = listenFeatures(samples, batches[0], augmenter: augmenter)
        var bi = 0
        while bi < batches.count {
            let current = prefetch.items
            let group = DispatchGroup()
            if bi + 1 < batches.count {
                let upcoming = batches[bi + 1]
                group.enter()
                DispatchQueue.global().async {
                    prefetch.items = listenFeatures(samples, upcoming, augmenter: augmenter)
                    group.leave()
                }
            }
            let (features, counts) = listenPadded(current)
            let (audio, _, kana) = listenAcoustic(acoustic, kanaVocabulary, features: features, counts: counts)
            var inputs: [MLXArray] = []
            if shape.kanaInput {
                var chunks: [Chunk] = []
                var i = 0
                while i < batches[bi].count {
                    let index = batches[bi][i]
                    let (chunk, recognized) = listenChunk(recognized: kana[i], reference: referenceKana[index], text: samples[index].text, vocabulary)
                    chunks.append(chunk)
                    if recognized {
                        recognizedUtterances += 1
                    }
                    i += 1
                }
                let sw = roundUp(chunks.map { $0.source.count }.max() ?? 1)
                let tw = roundUp(chunks.map { $0.target.count }.max() ?? 1)
                inputs = [
                    padded(chunks.map { $0.source }, width: sw, fill: KanaTextVocabulary.pad),
                    padded(chunks.map { $0.previous }, width: tw, fill: KanaTextVocabulary.pad),
                    padded(chunks.map { $0.target }, width: tw, fill: KanaTextVocabulary.pad),
                    unitMask(chunks, targetWidth: tw, sourceWidth: sw),
                    audio,
                    MLXKanaTextModel.paddingMask(counts, width: features.dim(1))
                ]
            } else {
                let targetsRows = batches[bi].map { vocabulary.encodeOutput(samples[$0].text) + [KanaTextVocabulary.eos] }
                let previousRows = targetsRows.map { [KanaTextVocabulary.bos] + $0.dropLast() }
                let tw = roundUp(targetsRows.map { $0.count }.max() ?? 1)
                inputs = [
                    MLXArray.zeros([1, 1], type: Int32.self),
                    padded(previousRows, width: tw, fill: KanaTextVocabulary.pad),
                    padded(targetsRows, width: tw, fill: KanaTextVocabulary.pad),
                    MLXArray.zeros([1, 1, 1, 1]),
                    audio,
                    MLXKanaTextModel.paddingMask(counts, width: features.dim(1))
                ]
            }
            step += 1
            // ArrayLRAdam has no bias correction; it is applied to the learning rate.
            let t = Float(step)
            let correction = (1.0 - pow(Float(0.98), t)).squareRoot() / (1.0 - pow(Float(0.9), t))
            optimizer.learningRate = MLXArray(correction * scheduler.learningRate(step: step, totalSteps: totalSteps, warmupSteps: ListenDefaults.warmupSteps))
            let (loss, grads) = lossAndGrad(model, inputs)
            let (clipped, _) = clipGradNorm(gradients: grads, maxNorm: Defaults.clipNorm)
            optimizer.update(model: model, gradients: clipped)
            eval(model, optimizer, loss)
            let value = loss[0].item(Float.self)
            lossSum += value
            windowSum += value
            if step % ListenDefaults.logEvery == 0 {
                print(String(format: "  step %d: loss %.4f (%.0f s into the epoch, peak %d MB)", step, windowSum / Float(ListenDefaults.logEvery),
                             CFAbsoluteTimeGetCurrent() - start, MLX.GPU.peakMemory >> 20))
                windowSum = 0.0
            }
            group.wait()
            bi += 1
        }
        print(String(format: "epoch %d/%d loss %.4f (%.0f s, peak %d MB)", ep, epochs, lossSum / Float(batches.count),
                     CFAbsoluteTimeGetCurrent() - start, MLX.GPU.peakMemory >> 20))
        if shape.kanaInput {
            print("  units from recognized kana in \(recognizedUtterances) of \(samples.count) utterances (the rest from reference kana)")
        }
        let selection = try evaluateAll(outDir + "/ep\(ep)")
        try saveModel(model, vocabulary: vocabulary, to: outDir + "/ep\(ep)")
        if 0.0 < selection && selection < best {
            best = selection
            try saveModel(model, vocabulary: vocabulary, to: outDir + "/best")
            print("  best so far (\(evalSets.first?.name ?? "") CER \(String(format: "%.2f", selection))%)")
        }
        ep += 1
    }
}
