import Foundation
import Haitaka

setbuf(stdout, nil)

// Evaluates kana-kanji conversion alone by feeding the reference kana; no acoustic model is needed.

var datasetPath = ""
var englishDictPath = ""
var dictSamples = 0        // lines used to build the dictionary and vocabulary (0 = all)
var showExamples = 3
var nbest = 0              // 0 disables N-best evaluation
var extraCorpusPath = ""   // extra text merged into the kana-kanji dictionary

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
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
    case "-s", "--samples":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                dictSamples = max(0, v)
            }
            argIdx += 1
        }
    case "--nbest":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                nbest = max(0, v)
            }
            argIdx += 1
        }
    case "--extra-corpus":
        if (argIdx + 1) < args.count {
            extraCorpusPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--examples":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                showExamples = max(0, v)
            }
            argIdx += 1
        }
    default:
        if arg.hasPrefix("-") != true {
            datasetPath = arg
        }
    }
    argIdx += 1
}

if datasetPath.isEmpty || englishDictPath.isEmpty {
    print("Error: specify a dataset directory and an English pronunciation dictionary.")
    print("  Usage: stage2 -d <dataset-dir> --english-dict <cmudict> [-s dict-lines] [--examples count]")
    exit(1)
}
guard let englishDict = try? EnglishPronunciations(contentsOfFile: englishDictPath) else {
    print("Error: cannot read pronunciation dictionary \(englishDictPath).")
    exit(1)
}

let transcriptPath = (datasetPath as NSString).appendingPathComponent("transcript_utf8.txt")
guard let content = try? String(contentsOfFile: transcriptPath, encoding: .utf8) else {
    print("Error: \(transcriptPath) not found.")
    exit(1)
}

var fileIds: [String] = []
var texts: [String] = []
for line in content.components(separatedBy: .newlines) {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty {
        continue
    }
    let parts = trimmed.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: true)
    if parts.count == 2 {
        fileIds.append(String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines))
        texts.append(String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

let dictLimit: Int
if dictSamples == 0 {
    dictLimit = texts.count
} else {
    dictLimit = min(dictSamples, texts.count)
}
let dictTexts = Array(texts.prefix(dictLimit))

let converter = KanjiConverter(english: englishDict)
let dictionary = KanaKanjiDictionary()
dictionary.buildFromCorpus(rawTexts: dictTexts, converter: converter)
if extraCorpusPath.isEmpty != true {
    let extraLines = DictionaryTextCorpus.load(path: extraCorpusPath)
    if extraLines.isEmpty {
        print("Failed to read extra corpus: \(extraCorpusPath)")
        exit(1)
    }
    dictionary.buildFromCorpus(rawTexts: extraLines, converter: converter)
    print("Extra corpus: \(extraCorpusPath) (\(extraLines.count) lines)")
}

print("Dataset: \(datasetPath)")
print("Lines: \(texts.count) total / \(dictTexts.count) used for the dictionary")
print("Dictionary entries: \(dictionary.count)")

func levenshtein(_ s1: String, _ s2: String) -> Int {
    let a = Array(s1)
    let b = Array(s2)
    if a.isEmpty { return b.count }
    if b.isEmpty { return a.count }
    var prev = Array(0...b.count)
    var cur = [Int](repeating: 0, count: b.count + 1)
    var i = 1
    while i <= a.count {
        cur[0] = i
        var j = 1
        while j <= b.count {
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            var best = prev[j] + 1
            if cur[j - 1] + 1 < best { best = cur[j - 1] + 1 }
            if prev[j - 1] + cost < best { best = prev[j - 1] + cost }
            cur[j] = best
            j += 1
        }
        prev = cur
        i += 1
    }
    return prev[b.count]
}

func stripPunctuation(_ text: String) -> String {
    var out = ""
    for c in text {
        switch c {
        case "、", "。", "，", "．", "・":
            break
        default:
            out.append(c)
        }
    }
    return out
}

struct Result {
    let fileId: String
    let target: String
    let produced: String
    let cer: Float
    let cerNoPunct: Float
    let isExact: Bool
    let isExactNoPunct: Bool
    let isDictSource: Bool
}

let decoder = KanaKanjiDecoder(dictionary: dictionary)
var results: [Result] = []
let start = Date()

var idx = 0
while idx < texts.count {
    let target = texts[idx]
    let kana = converter.convertToHiragana(target)
    let produced = decoder.decode(kanaText: kana)

    let dist = levenshtein(target, produced)
    let cer = Float(dist) / Float(max(1, target.count))
    let tNo = stripPunctuation(target)
    let pNo = stripPunctuation(produced)
    let distNo = levenshtein(tNo, pNo)
    let cerNo = Float(distNo) / Float(max(1, tNo.count))

    results.append(Result(
        fileId: fileIds[idx],
        target: target,
        produced: produced,
        cer: cer,
        cerNoPunct: cerNo,
        isExact: target == produced,
        isExactNoPunct: tNo == pNo,
        isDictSource: idx < dictLimit
    ))
    idx += 1
}

let elapsed = Date().timeIntervalSince(start)

func summarize(_ list: [Result], _ label: String) {
    if list.isEmpty {
        return
    }
    let n = Float(list.count)
    var sumCer: Float = 0.0
    var sumCerNo: Float = 0.0
    var exact = 0
    var exactNo = 0
    for r in list {
        sumCer += r.cer
        sumCerNo += r.cerNoPunct
        if r.isExact { exact += 1 }
        if r.isExactNoPunct { exactNo += 1 }
    }
    print("[\(label)] \(list.count) lines")
    print("  with punct.     CER: \(String(format: "%.2f", sumCer / n * 100.0))%  exact: \(String(format: "%.1f", Float(exact) / n * 100.0))%")
    print("  without punct.  CER: \(String(format: "%.2f", sumCerNo / n * 100.0))%  exact: \(String(format: "%.1f", Float(exactNo) / n * 100.0))%")
}

print("\n==================================================")
print("=== Stage 2 alone (reference kana input) ===")
print("==================================================")
summarize(results.filter { $0.isDictSource }, "lines used for the dictionary")
let unseen = results.filter { $0.isDictSource != true }
summarize(unseen, "lines not used for the dictionary")
print("\nElapsed: \(String(format: "%.2f", elapsed)) s (\(String(format: "%.1f", Double(results.count) / elapsed)) lines/s)")

// N-best oracle: upper bound if the candidate closest to the reference were always chosen.
if 0 < nbest {
    let nbStart = Date()
    var sumOracle: Float = 0.0
    var oracleExact = 0
    var sumTop1: Float = 0.0
    var rankSum = 0
    var evaluated = 0

    var i2 = 0
    while i2 < texts.count {
        if i2 < dictLimit {
            let target = stripPunctuation(texts[i2])
            let kana = converter.convertToHiragana(texts[i2])
            let cands = decoder.decodeNBest(kanaText: kana, beamWidth: nbest)
            if cands.isEmpty != true {
                var bestCer = Float.greatestFiniteMagnitude
                var bestRank = 0
                var r = 0
                while r < cands.count {
                    let p = stripPunctuation(cands[r].text)
                    let c = Float(levenshtein(target, p)) / Float(max(1, target.count))
                    if c < bestCer {
                        bestCer = c
                        bestRank = r
                    }
                    r += 1
                }
                let top1 = stripPunctuation(cands[0].text)
                sumTop1 += Float(levenshtein(target, top1)) / Float(max(1, target.count))
                sumOracle += bestCer
                if bestCer == 0.0 {
                    oracleExact += 1
                }
                rankSum += bestRank
                evaluated += 1
            }
        }
        i2 += 1
    }

    let nf = Float(max(1, evaluated))
    print("\n==================================================")
    print("=== N-best oracle (K = \(nbest), without punctuation) ===")
    print("==================================================")
    print("  Top-1     CER: \(String(format: "%.2f", sumTop1 / nf * 100.0))%")
    print("  oracle  CER: \(String(format: "%.2f", sumOracle / nf * 100.0))%  exact: \(String(format: "%.1f", Float(oracleExact) / nf * 100.0))%")
    print("  mean rank of the best candidate: \(String(format: "%.2f", Float(rankSum) / nf))")
    print("  elapsed: \(String(format: "%.2f", Date().timeIntervalSince(nbStart))) s")
}

if 0 < showExamples {
    // Examples that match exactly except for punctuation.
    let punctOnly = results.filter { $0.isDictSource && $0.isExactNoPunct && ($0.isExact != true) }
    print("\n--- Punctuation-only errors: \(punctOnly.count) (of lines used for the dictionary: \(String(format: "%.1f", Float(punctOnly.count) * 100.0 / Float(max(1, results.filter { $0.isDictSource }.count))))%) ---")
    for r in punctOnly.prefix(showExamples * 3) {
        print("  reference: \"\(r.target)\"")
        print("  output:    \"\(r.produced)\"")
    }

    let worst = results
        .filter { $0.isDictSource }
        .sorted { b, a in a.cerNoPunct < b.cerNoPunct }
        .prefix(showExamples)

    print("\n--- Worst \(worst.count) (lines used for the dictionary) ---")
    for r in worst {
        print("\n[\(r.fileId)] CER without punct.: \(String(format: "%.1f", r.cerNoPunct * 100.0))%")
        print("  reference kana:  \"\(converter.convertToHiragana(r.target))\"")
        print("  reference kanji: \"\(r.target)\"")
        print("  output:          \"\(r.produced)\"")
        let traceDecoder = KanaKanjiDecoder(dictionary: dictionary)
        _ = traceDecoder.decode(kanaText: converter.convertToHiragana(r.target))
        for seg in traceDecoder.lastTrace {
            print("    \(seg.kanaRange) → \(seg.emitted)  [\(seg.kind)] \(String(format: "%+.2f", seg.stepScore))")
        }
    }
}
