import Foundation
import Haitaka

setbuf(stdout, nil)

// Cuts long audio into utterances, attaches transcript text to each, and writes a training manifest.
//
// Two transcript formats are accepted:
//   Timed: SRT (body is the lines after `HH:MM:SS,mmm --> HH:MM:SS,mmm` up to a blank line).
//          Cut at the cue times; only cues longer than --max-seconds are split by VAD with sentences assigned.
//   Plain text: segments are cut at VAD silences (SpeechChunker, as in transcribe); sentences end at
//          sentence-final punctuation (period, !, ?, ellipsis) and are assigned to segments by dynamic
//          programming so that the seconds-to-characters ratio is uniform. This is not forced
//          alignment with the acoustic model, so segments with outlying characters/second are
//          reported for review.
//
// Usage: segment -i <audio.wav> -t <transcript.srt | .txt | string> -o <output dir> --english-dict <cmudict> [--max-seconds 15]
//        Writes {"path","text"} JSONL to stdout.

/// Max silence (seconds) merged into one utterance.
let mergeGapSeconds: Float = 0.35
/// Segments shorter than this (seconds) are dropped.
let minSegmentSeconds: Float = 0.6
/// Max segment length (seconds). Longer segments are split at their quietest window.
var maxSegmentSeconds: Float = 15.0
/// Character density (chars/second) outside this range is reported for review after assignment.
let plausibleCharsPerSecond: ClosedRange<Float> = 3.0...12.0
let sampleRate: Float = 16000.0

var wavPath = ""
var textArg = ""
var outDir = ""
var englishDictPath = ""
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    switch args[argIdx] {
    case "-i", "--input":
        if (argIdx + 1) < args.count {
            wavPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-t", "--text":
        if (argIdx + 1) < args.count {
            textArg = args[argIdx + 1]
            argIdx += 1
        }
    case "-o", "--output":
        if (argIdx + 1) < args.count {
            outDir = args[argIdx + 1]
            argIdx += 1
        }
    case "--english-dict":
        if (argIdx + 1) < args.count {
            englishDictPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--max-seconds":
        if (argIdx + 1) < args.count {
            if let v = Float(args[argIdx + 1]) {
                maxSegmentSeconds = max(2.0, v)
            }
            argIdx += 1
        }
    default:
        break
    }
    argIdx += 1
}

func warn(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

if wavPath.isEmpty || textArg.isEmpty || outDir.isEmpty || englishDictPath.isEmpty {
    warn("Usage: segment -i <audio.wav> -t <transcript> -o <output-dir> --english-dict <cmudict> [--max-seconds 15]")
    exit(1)
}
guard let englishDict = try? EnglishPronunciations(contentsOfFile: englishDictPath) else {
    warn("Error: cannot read pronunciation dictionary \(englishDictPath).")
    exit(1)
}

// The transcript may be a file path or the text itself.
var transcript = textArg
if FileManager.default.fileExists(atPath: textArg) {
    guard let t = try? String(contentsOfFile: textArg, encoding: .utf8) else {
        warn("Error: cannot read transcript: \(textArg)")
        exit(1)
    }
    transcript = t
}

guard let wav = SpeechDataset.loadWavFile(path: wavPath) else {
    warn("Error: cannot read audio: \(wavPath)")
    exit(1)
}
let pcm = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
let totalSeconds = Float(pcm.count) / sampleRate
let converter = KanjiConverter(english: englishDict)
let chunker = SpeechChunker(
    mergeGapSeconds: mergeGapSeconds,
    minSegmentSeconds: minSegmentSeconds,
    maxSegmentSeconds: maxSegmentSeconds
)

extension SpeechChunker.Span {
    var seconds: Float {
        return Float(end - start) / 16000.0
    }
}

// MARK: - Sentence splitting and assignment

/// Sentence-ending characters. The character stays with the preceding sentence.
let sentenceBreaks: Set<Character> = ["。", "！", "？", "!", "?", "…", "\n"]

func splitSentences(_ text: String, atLeast minimum: Int) -> [String] {
    var sentences: [String] = []
    var current = ""
    for ch in text {
        current.append(ch)
        if sentenceBreaks.contains(ch) {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty != true {
                sentences.append(trimmed)
            }
            current = ""
        }
    }
    let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
    if tail.isEmpty != true {
        sentences.append(tail)
    }
    // With fewer sentences than segments, also split at commas.
    if sentences.count < minimum {
        var finer: [String] = []
        for s in sentences {
            var buf = ""
            for ch in s {
                buf.append(ch)
                if ch == "、" || ch == "," {
                    finer.append(buf)
                    buf = ""
                }
            }
            if buf.isEmpty != true {
                finer.append(buf)
            }
        }
        sentences = finer
    }
    return sentences
}

/// Length in kana, excluding symbols and whitespace, used for density.
func speechLength(_ s: String) -> Int {
    return converter.kanaOnly(converter.convertToHiragana(s)).count
}

/// Assigns sentences to spans in order. With more spans than sentences, short spans are merged with neighbors.
/// DP cost of placing sentences i..<j in span k = (chars - spanSeconds * overall chars/sec)^2 / spanSeconds.
func assign(sentences: [String], to inputSpans: [SpeechChunker.Span]) -> [(span: SpeechChunker.Span, text: String, chars: Int)] {
    var spans = inputSpans
    let sCount = sentences.count
    if sCount == 0 || spans.isEmpty {
        return []
    }
    if sCount < spans.count {
        warn("Warning: fewer sentences (\(sCount)) than spans (\(spans.count)); merging spans to match the sentences")
    }
    let kCount = min(spans.count, sCount)
    while kCount < spans.count {
        var minIdx = 0
        var minLen = Float.greatestFiniteMagnitude
        var idx = 0
        while idx < spans.count {
            if spans[idx].seconds < minLen {
                minLen = spans[idx].seconds
                minIdx = idx
            }
            idx += 1
        }
        let neighbor: Int
        if minIdx == 0 {
            neighbor = 1
        } else {
            neighbor = minIdx - 1
        }
        let lo = min(minIdx, neighbor)
        let hi = max(minIdx, neighbor)
        spans[lo].end = spans[hi].end
        spans.remove(at: hi)
    }

    let lengths = sentences.map { speechLength($0) }
    var prefix = [Int](repeating: 0, count: sCount + 1)
    var i = 0
    while i < sCount {
        prefix[i + 1] = prefix[i] + lengths[i]
        i += 1
    }
    let totalSpeech = spans.reduce(Float(0.0)) { $0 + $1.seconds }
    let targetRate = Float(prefix[sCount]) / max(0.1, totalSpeech)

    let inf = Float.greatestFiniteMagnitude
    var cost = [[Float]](repeating: [Float](repeating: inf, count: sCount + 1), count: kCount + 1)
    var choice = [[Int]](repeating: [Int](repeating: 0, count: sCount + 1), count: kCount + 1)
    cost[0][0] = 0.0
    var k = 1
    while k <= kCount {
        let seconds = spans[k - 1].seconds
        let expected = seconds * targetRate
        var j = k
        while j <= sCount - (kCount - k) {
            var best = inf
            var bestI = 0
            var ii = k - 1
            while ii < j {
                if cost[k - 1][ii] < inf {
                    let diff = Float(prefix[j] - prefix[ii]) - expected
                    let c = cost[k - 1][ii] + (diff * diff) / max(0.5, seconds)
                    if c < best {
                        best = c
                        bestI = ii
                    }
                }
                ii += 1
            }
            cost[k][j] = best
            choice[k][j] = bestI
            j += 1
        }
        k += 1
    }
    var boundaries = [Int](repeating: 0, count: kCount + 1)
    boundaries[kCount] = sCount
    k = kCount
    while 0 < k {
        boundaries[k - 1] = choice[k][boundaries[k]]
        k -= 1
    }

    var result: [(span: SpeechChunker.Span, text: String, chars: Int)] = []
    k = 0
    while k < kCount {
        result.append((
            span: spans[k],
            text: sentences[boundaries[k]..<boundaries[k + 1]].joined(),
            chars: prefix[boundaries[k + 1]] - prefix[boundaries[k]]
        ))
        k += 1
    }
    return result
}

// MARK: - Segment boundaries

/// One SRT cue.
struct TimedLine {
    let start: Float
    let end: Float
    let text: String
}

/// Parses `HH:MM:SS,mmm` (separator `,` or `.`) to seconds; nil if malformed.
func parseSRTTime(_ s: String) -> Float? {
    let t = s.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
    let parts = t.components(separatedBy: ":")
    if parts.count != 3 {
        return nil
    }
    guard let h = Float(parts[0]), let m = Float(parts[1]), let sec = Float(parts[2]) else {
        return nil
    }
    return (h * 3600.0) + (m * 60.0) + sec
}

/// Parses SRT. Index lines are ignored; the text runs from the line after the timing line to a blank line (multiple lines are joined).
func parseSRT(_ rawText: String) -> [TimedLine] {
    // Normalize CRLF and BOM first; they break cue detection and time parsing.
    let text = rawText
        .replacingOccurrences(of: "\u{FEFF}", with: "")
        .replacingOccurrences(of: "\r\n", with: "\n")
        .replacingOccurrences(of: "\r", with: "\n")
    var cues: [TimedLine] = []
    var pendingStart: Float = 0.0
    var pendingEnd: Float = 0.0
    var inCue = false
    var body = ""

    func closeCue() {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if inCue && trimmed.isEmpty != true && pendingStart < pendingEnd {
            cues.append(TimedLine(start: pendingStart, end: pendingEnd, text: trimmed))
        }
        inCue = false
        body = ""
    }

    for raw in text.components(separatedBy: "\n") {
        let line = raw.trimmingCharacters(in: .whitespaces)
        if line.isEmpty {
            closeCue()
            continue
        }
        if line.contains("-->") {
            closeCue()
            let parts = line.components(separatedBy: "-->")
            if parts.count == 2, let s = parseSRTTime(parts[0]), let e = parseSRTTime(parts[1]) {
                pendingStart = s
                pendingEnd = e
                inCue = true
            }
            continue
        }
        if inCue {
            body += line
        }
    }
    closeCue()
    return cues
}

/// `exact` is true for SRT cue timing (alignment exact), false for VAD splits with assigned sentences (alignment estimated).
var pieces: [(span: SpeechChunker.Span, text: String, chars: Int, exact: Bool)] = []
let timed = parseSRT(transcript)
let modeDescription: String
if timed.isEmpty {
    modeDescription = "VAD split + sentence assignment"
    let spans = chunker.chunk(pcm: pcm)
    if spans.isEmpty {
        warn("Error: no voiced span of sufficient length")
        exit(1)
    }
    pieces = assign(sentences: splitSentences(transcript, atLeast: spans.count), to: spans).map {
        (span: $0.span, text: $0.text, chars: $0.chars, exact: false)
    }
} else {
    modeDescription = "SRT"
    var skipped = 0
    for line in timed {
        let start = max(0, Int(line.start * sampleRate))
        let end = min(pcm.count, Int(line.end * sampleRate))
        if end <= start {
            skipped += 1
            continue
        }
        let span = SpeechChunker.Span(start: start, end: end)
        if span.seconds <= maxSegmentSeconds {
            pieces.append((span: span, text: line.text, chars: speechLength(line.text), exact: true))
            continue
        }
        // Only cues over the limit are split by VAD with sentences assigned.
        let sub = chunker.chunk(pcm: Array(pcm[start..<end])).map {
            SpeechChunker.Span(start: $0.start + start, end: $0.end + start)
        }
        if sub.isEmpty {
            pieces.append((span: span, text: line.text, chars: speechLength(line.text), exact: true))
            continue
        }
        pieces += assign(sentences: splitSentences(line.text, atLeast: sub.count), to: sub).map {
            (span: $0.span, text: $0.text, chars: $0.chars, exact: false)
        }
    }
    if 0 < skipped {
        warn("Warning: skipped \(skipped) lines (outside the audio, etc.)")
    }
}

// MARK: - WAV and manifest output

func writeWav(_ samples: ArraySlice<Float>, to path: String) -> Bool {
    var data = Data()
    let dataBytes = UInt32(samples.count * 2)
    func appendU32(_ v: UInt32) {
        var le = v.littleEndian
        data.append(Data(bytes: &le, count: 4))
    }
    func appendU16(_ v: UInt16) {
        var le = v.littleEndian
        data.append(Data(bytes: &le, count: 2))
    }
    data.append("RIFF".data(using: .ascii)!)
    appendU32(36 + dataBytes)
    data.append("WAVE".data(using: .ascii)!)
    data.append("fmt ".data(using: .ascii)!)
    appendU32(16)
    appendU16(1)
    appendU16(1)
    appendU32(16000)
    appendU32(16000 * 2)
    appendU16(2)
    appendU16(16)
    data.append("data".data(using: .ascii)!)
    appendU32(dataBytes)
    var pcm16 = [Int16](repeating: 0, count: samples.count)
    var n = 0
    for v in samples {
        let clamped = max(-1.0, min(1.0, v))
        pcm16[n] = Int16(clamped * 32767.0)
        n += 1
    }
    pcm16.withUnsafeBufferPointer { buf in
        data.append(Data(buffer: buf))
    }
    do {
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        return true
    } catch {
        return false
    }
}

try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
let baseName = ((wavPath as NSString).lastPathComponent as NSString).deletingPathExtension
var flagged = 0
var totalChars = 0
var totalSpeech: Float = 0.0
var index = 0
while index < pieces.count {
    let piece = pieces[index]
    let rate = Float(piece.chars) / piece.span.seconds
    let outPath = (outDir as NSString).appendingPathComponent(String(format: "%@_seg%03d.wav", baseName, index + 1))
    if writeWav(pcm[piece.span.start..<piece.span.end], to: outPath) != true {
        warn("Error: failed to write: \(outPath)")
        exit(1)
    }
    let escaped = piece.text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    // `source` and `cps` (chars/second) let callers drop poorly aligned segments; training reads only path and text.
    var source = "vad"
    if piece.exact {
        source = "cue"
    }
    print(String(format: "{\"path\": \"%@\", \"text\": \"%@\", \"source\": \"%@\", \"cps\": %.2f}", outPath, escaped, source, rate))
    if piece.exact != true && plausibleCharsPerSecond.contains(rate) != true {
        flagged += 1
        warn(String(format: "  CHECK seg%03d [%.1f -> %.1f]: %d chars = %.1f chars/s: %@",
                    index + 1, Float(piece.span.start) / sampleRate, Float(piece.span.end) / sampleRate,
                    piece.chars, rate, String(piece.text.prefix(40))))
    }
    totalChars += piece.chars
    totalSpeech += piece.span.seconds
    index += 1
}
warn(String(format: "Done (%@): split %.1f s of audio into %d segments (overall %.1f chars/s, %d to check)",
            modeDescription, totalSeconds, pieces.count, Float(totalChars) / max(0.1, totalSpeech), flagged))
