import Foundation

/// Characters treated as punctuation in kana-kanji conversion.
let punctuationCharacters: Set<Character> = ["、", "。", "，", "．"]

/// Dictionary entry: a surface form with its reading.
public struct KanaKanjiEntry: Sendable {
    public let reading: String       // hiragana reading, e.g. "とつぜん", "ぱにくって"
    public let surface: String       // surface form, e.g. "突然", "パニクって"
    public let frequency: Int        // occurrence count
    public let isParticle: Bool      // particle or sentence-final expression

    public init(reading: String, surface: String, frequency: Int = 1, isParticle: Bool = false) {
        self.reading = reading
        self.surface = surface
        self.frequency = frequency
        self.isParticle = isParticle
    }
}

/// Word dictionary for kana-kanji conversion.
///
/// Scores are log-probabilities: P(reading | surface) from pair counts and
/// P(surface | previous surface) from word bigram counts.
public final class KanaKanjiDictionary: @unchecked Sendable {
    /// Bigram/unigram interpolation weight (1.0 = bigram only).
    private let bigramInterpolation: Float = 0.7

    private var entriesByReading: [String: [KanaKanjiEntry]] = [:]
    private var readingTotals: [String: Int] = [:]
    private var surfaceTotals: [String: Int] = [:]
    private var bigramCounts: [String: [String: Int]] = [:]
    private var vocabularySize: Int = 0
    private var readingsByLength: [Int: [String]] = [:]
    /// Readings keyed by each one-character-wildcard variant. Below distance 0.8 the only candidates are
    /// same-length readings differing in one character, so they are found by hash lookups without a scan.
    private var oneSubstitutionIndex: [String: [String]] = [:]
    private var totalTokens: Int = 0
    /// Probability that a word is followed by "、". The acoustic model emits no punctuation, so it is restored from corpus statistics.
    public private(set) var commaAfterRate: [String: Float] = [:]
    /// Most frequent sentence-final punctuation and its rate.
    public private(set) var sentenceFinalPunctuation: Character? = nil
    public private(set) var sentenceFinalRate: Float = 0.0

    public init(includeSeed: Bool = true) {
        if includeSeed {
            loadSeedVocabulary()
        }
    }

    /// Adds the built-in `SeedVocabulary` entries.
    public func loadSeedVocabulary() {
        for entry in SeedVocabulary.entries {
            addEntry(entry)
        }
        vocabularySize = surfaceTotals.count
    }

    public var count: Int {
        return entriesByReading.count
    }

    public var allReadings: [String] {
        return Array(entriesByReading.keys)
    }

    /// Adds an entry; frequencies of an existing (reading, surface) pair are summed.
    public func addEntry(_ entry: KanaKanjiEntry) {
        // Normalize readings to the pronunciation kana the acoustic model emits (えいち -> えーち),
        // otherwise its output would not match the dictionary keys.
        let reading = KanjiConverter.pronunciation(surface: entry.surface, reading: entry.reading)
        var list = entriesByReading[reading] ?? []

        var found = false
        var i = 0
        while i < list.count {
            if list[i].surface == entry.surface {
                list[i] = KanaKanjiEntry(
                    reading: reading,
                    surface: entry.surface,
                    frequency: list[i].frequency + entry.frequency,
                    isParticle: list[i].isParticle
                )
                found = true
                break
            }
            i += 1
        }
        if found != true {
            list.append(entry)
        }

        if entriesByReading[reading] == nil {
            readingsByLength[reading.count, default: []].append(reading)
            for key in Self.wildcardKeys(for: reading) {
                oneSubstitutionIndex[key, default: []].append(reading)
            }
        }
        entriesByReading[reading] = list
        readingTotals[reading, default: 0] += entry.frequency
        surfaceTotals[entry.surface, default: 0] += entry.frequency
        totalTokens += entry.frequency
    }

    public func addBigram(from: String, to: String) {
        var map = bigramCounts[from] ?? [:]
        map[to, default: 0] += 1
        bigramCounts[from] = map
    }

    /// Keys with each character in turn replaced by "*".
    static func wildcardKeys(for reading: String) -> [String] {
        let chars = Array(reading)
        var keys: [String] = []
        keys.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            var key = ""
            var j = 0
            while j < chars.count {
                if j == i {
                    key.append("*")
                } else {
                    key.append(chars[j])
                }
                j += 1
            }
            keys.append(key)
            i += 1
        }
        return keys
    }

    /// log P(reading | surface). Usually near 0, since a surface form rarely has several readings.
    public func logReadingGivenSurface(reading: String, surface: String) -> Float {
        let surfaceTotal = surfaceTotals[surface] ?? 0
        if surfaceTotal <= 0 {
            return -.infinity
        }
        var pairCount = 0
        for e in entriesByReading[reading] ?? [] {
            if e.surface == surface {
                pairCount = e.frequency
                break
            }
        }
        if pairCount <= 0 {
            return -.infinity
        }
        return log(Float(pairCount) / Float(surfaceTotal))
    }

    /// log P(surface | previous surface), interpolating bigram and unigram so unseen bigrams
    /// still differ by unigram frequency.
    public func logBigram(from: String, to: String) -> Float {
        let total = Float(max(1, totalTokens))
        let unigram = Float(surfaceTotals[to] ?? 0) / total

        let prevTotal = Float(surfaceTotals[from] ?? 0)
        var bigram: Float = 0.0
        if 0 < prevTotal {
            bigram = Float(bigramCounts[from]?[to] ?? 0) / prevTotal
        }

        let mixed = (bigramInterpolation * bigram) + ((1.0 - bigramInterpolation) * unigram)
        if mixed <= 0.0 {
            // Word not in the dictionary.
            return log(1.0 / (total * total))
        }
        return log(mixed)
    }

    public func lookupExact(reading: String) -> [KanaKanjiEntry] {
        return entriesByReading[reading] ?? []
    }

    /// Builds entries, word bigrams and punctuation statistics from transcripts.
    ///
    /// Surfaces and readings come from the same tokenization; readings estimated separately would
    /// not match the kana the acoustic model emits.
    public func buildFromCorpus(rawTexts: [String], converter: KanjiConverter = KanjiConverter()) {
        var commaFollows: [String: Int] = [:]
        var wordOccurrences: [String: Int] = [:]
        var finalPunctCounts: [Character: Int] = [:]
        var sentenceCount = 0

        for text in rawTexts {
            var prevWord: String? = nil

            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty != true {
                sentenceCount += 1
                if let last = trimmed.last {
                    if punctuationCharacters.contains(last) {
                        finalPunctCounts[last, default: 0] += 1
                    }
                }
            }

            // Count "、" directly after each word.
            var lastContentWord: String? = nil
            for token in converter.tokenize(text) {
                let surface = token.surface.trimmingCharacters(in: .whitespacesAndNewlines)
                if surface.isEmpty {
                    continue
                }

                // Punctuation tokens have no reading; record them only as following the previous word.
                if surface.count == 1 {
                    if let ch = surface.first {
                        if punctuationCharacters.contains(ch) {
                            switch lastContentWord {
                            case .some(let w):
                                if ch == "、" {
                                    commaFollows[w, default: 0] += 1
                                }
                            case .none:
                                break
                            }
                            continue
                        }
                    }
                }

                let reading = token.reading
                if reading.isEmpty {
                    continue
                }

                addEntry(KanaKanjiEntry(reading: reading, surface: surface, frequency: 1))
                // Also index colloquial readings of the same surface (あたし -> 私).
                for variant in KanjiConverter.readingVariants[surface] ?? [] {
                    addEntry(KanaKanjiEntry(reading: variant, surface: surface, frequency: 1))
                }
                wordOccurrences[surface, default: 0] += 1
                lastContentWord = surface

                switch prevWord {
                case .some(let pWord):
                    addBigram(from: pWord, to: surface)
                case .none:
                    ()
                }

                prevWord = surface
            }
        }

        var bestFinal: Character? = nil
        var bestFinalCount = 0
        for (ch, cnt) in finalPunctCounts {
            if bestFinalCount < cnt {
                bestFinalCount = cnt
                bestFinal = ch
            }
        }
        sentenceFinalPunctuation = bestFinal
        if 0 < sentenceCount {
            sentenceFinalRate = Float(bestFinalCount) / Float(sentenceCount)
        }

        for (w, cnt) in commaFollows {
            let total = wordOccurrences[w] ?? 0
            if 0 < total {
                commaAfterRate[w] = Float(cnt) / Float(total)
            }
        }

        vocabularySize = surfaceTotals.count
    }


    /// Fuzzy lookup by articulatory edit distance (`phoneticDistance`).
    public func lookupFuzzyPhonetic(reading: String, maxPhoneticDist: Float = 1.25) -> [(entry: KanaKanjiEntry, dist: Float)] {
        var results: [(entry: KanaKanjiEntry, dist: Float)] = []
        let queryChars = Array(reading)
        let rLen = queryChars.count

        // Insertion and deletion cost 0.8 per character, so below 0.8 only same-length readings with
        // one substitution qualify; fetch them from the wildcard index.
        if maxPhoneticDist < 0.8 {
            var seen = Set<String>()
            for key in Self.wildcardKeys(for: reading) {
                for candReading in oneSubstitutionIndex[key] ?? [] {
                    if candReading == reading || seen.contains(candReading) {
                        continue
                    }
                    seen.insert(candReading)
                    let dist = diagonalDistance(queryChars, Array(candReading), limit: maxPhoneticDist)
                    if dist <= maxPhoneticDist {
                        for e in entriesByReading[candReading] ?? [] {
                            results.append((entry: e, dist: dist))
                        }
                    }
                }
            }
            return results
        }

        // Otherwise scan readings within the reachable length range.
        let maxLengthDiff = Int(maxPhoneticDist / 0.8)
        var length = rLen - maxLengthDiff
        while length <= rLen + maxLengthDiff {
            if length <= 0 {
                length += 1
                continue
            }
            for candReading in readingsByLength[length] ?? [] {
                let dist = phoneticDistance(reading, candReading)
                if dist <= maxPhoneticDist {
                    for e in entriesByReading[candReading] ?? [] {
                        results.append((entry: e, dist: dist))
                    }
                }
            }
            length += 1
        }

        return results
    }

    /// Substitution-only distance between same-length readings; returns infinity once `limit` is exceeded.
    /// Exact when `limit` < 1.6, since any non-diagonal alignment needs an insertion and a deletion (0.8 each).
    private func diagonalDistance(_ a: [Character], _ b: [Character], limit: Float) -> Float {
        var total: Float = 0.0
        var i = 0
        while i < a.count {
            if a[i] != b[i] {
                total += charSubstitutionCost(a[i], b[i])
                if limit < total {
                    return .infinity
                }
            }
            i += 1
        }
        return total
    }

    /// Edit distance with insertion/deletion 0.8 and substitution costs from articulatory similarity.
    public func phoneticDistance(_ s1: String, _ s2: String) -> Float {
        let a1 = Array(s1)
        let a2 = Array(s2)
        let m = a1.count
        let n = a2.count
        if m == 0 { return Float(n) * 0.8 }
        if n == 0 { return Float(m) * 0.8 }

        var dp = [[Float]](repeating: [Float](repeating: 0.0, count: n + 1), count: m + 1)
        var i = 0
        while i <= m {
            dp[i][0] = Float(i) * 0.8
            i += 1
        }
        var j = 0
        while j <= n {
            dp[0][j] = Float(j) * 0.8
            j += 1
        }

        i = 1
        while i <= m {
            j = 1
            while j <= n {
                let c1 = a1[i - 1]
                let c2 = a2[j - 1]
                let substCost = charSubstitutionCost(c1, c2)

                let d1 = dp[i - 1][j] + 0.8
                let d2 = dp[i][j - 1] + 0.8
                let d3 = dp[i - 1][j - 1] + substCost

                var minVal = d1
                if d2 < minVal { minVal = d2 }
                if d3 < minVal { minVal = d3 }
                dp[i][j] = minVal
                j += 1
            }
            i += 1
        }
        return dp[m][n]
    }

    // MARK: - Articulatory similarity tables
    private static let voicedPairs: Set<String> = [
        "かが", "きぎ", "くぐ", "けげ", "こご",
        "さざ", "しじ", "すず", "せぜ", "そぞ",
        "ただ", "ちぢ", "つづ", "てで", "とど",
        "はば", "ひび", "ふぶ", "へべ", "ほぼ",
        "はぱ", "ひぴ", "ふぷ", "へぺ", "ほぽ",
        "ばぱ", "びぴ", "ぶぷ", "べぺ", "ぼぽ"
    ]
    private static let vowels: Set<Character> = ["あ", "い", "う", "え", "お"]
    private static let specialPairs: Set<String> = [
        "ーう", "ーお", "ーい", "ーえ", "ーあ",
        "っつ", "んむ", "んう"
    ]
    private static let vowelPairs: Set<String> = ["おう", "うお", "えい", "いえ", "あお", "おあ"]

    /// Substitution cost between two kana by articulatory similarity.
    private func charSubstitutionCost(_ c1: Character, _ c2: Character) -> Float {
        if c1 == c2 {
            return 0.0
        }

        // Voiced/unvoiced/semi-voiced pairs (か/が, は/ぱ).
        let pairStr = "\(c1)\(c2)"
        let revPairStr = "\(c2)\(c1)"

        if Self.voicedPairs.contains(pairStr) || Self.voicedPairs.contains(revPairStr) {
            return 0.35
        }

        // Long vowel vs. vowel, geminate (っ/つ) and moraic nasal (ん/む) variation.
        let isProlongedVowel = (c1 == "ー" && Self.vowels.contains(c2)) || (c2 == "ー" && Self.vowels.contains(c1))
        if isProlongedVowel || Self.specialPairs.contains(pairStr) || Self.specialPairs.contains(revPairStr) {
            return 0.40
        }

        // Similar vowel sequences (おう/うお, えい/いえ).
        if Self.vowelPairs.contains(pairStr) || Self.vowelPairs.contains(revPairStr) {
            return 0.45
        }

        return 1.0
    }
}

/// Converts pronunciation kana to kanji-kana text by beam Viterbi over dictionary words,
/// scored by reading likelihood, word bigrams and fuzzy phonetic matches.
public final class KanaKanjiDecoder: @unchecked Sendable {
    /// Longest reading looked up, in characters.
    private static let maxWordLength = 12
    /// Shortest reading allowed to match fuzzily.
    private static let minFuzzyLength = 3
    private static let maxFuzzyDistance: Float = 0.6
    /// Log-probability per unit of phonetic distance; large so exact matches clearly win.
    private static let fuzzyDistanceCost: Float = -12.0
    /// Log-probability for passing one unknown kana through unchanged.
    private static let unknownCharLogProb: Float = -12.0
    /// States kept per position. Bigram scores depend on the previous word, so one state per position is not exact.
    private static let defaultBeamWidth = 5

    public let dictionary: KanaKanjiDictionary
    public let bigramWeight: Float
    /// Minimum `commaAfterRate` to insert "、" after a word.
    public let commaThreshold: Float
    /// Minimum `sentenceFinalRate` to append sentence-final punctuation.
    public let sentenceFinalThreshold: Float

    /// Segments of the best path from the last decode, for diagnostics.
    public private(set) var lastTrace: [TraceSegment] = []

    public init(
        dictionary: KanaKanjiDictionary,
        bigramWeight: Float = 1.0,
        commaThreshold: Float = 0.5,
        sentenceFinalThreshold: Float = 0.5
    ) {
        self.dictionary = dictionary
        self.bigramWeight = bigramWeight
        self.commaThreshold = commaThreshold
        self.sentenceFinalThreshold = sentenceFinalThreshold
    }

    public struct TraceSegment: Sendable {
        public let kanaRange: String
        public let emitted: String
        public let kind: String
        public let stepScore: Float
    }

    private func commaSuffix(after surface: String) -> String {
        switch dictionary.commaAfterRate[surface] {
        case .some(let rate):
            if commaThreshold <= rate {
                return "、"
            }
            return ""
        case .none:
            return ""
        }
    }

    private func sentenceFinalSuffix() -> String {
        switch dictionary.sentenceFinalPunctuation {
        case .some(let ch):
            if sentenceFinalThreshold <= dictionary.sentenceFinalRate {
                return String(ch)
            }
            return ""
        case .none:
            return ""
        }
    }

    public struct Candidate: Sendable {
        public let text: String
        public let score: Float
    }

    /// Returns up to `beamWidth` candidates, best first. With `beamWidth` 1 this is plain Viterbi.
    public func decodeNBest(kanaText: String, beamWidth: Int) -> [Candidate] {
        if kanaText.isEmpty {
            return []
        }
        let k = max(1, beamWidth)
        let chars = Array(kanaText)
        let n = chars.count

        struct BeamState {
            var score: Float
            var text: String
            var lastWord: String
            var trace: [TraceSegment]
        }

        var beams = [[BeamState]](repeating: [], count: n + 1)
        beams[0] = [BeamState(score: 0.0, text: "", lastWord: "", trace: [])]

        func push(_ index: Int, _ state: BeamState) {
            var list = beams[index]
            list.append(state)
            list.sort { b, a in a.score < b.score }
            if k < list.count {
                list.removeLast(list.count - k)
            }
            beams[index] = list
        }

        var i = 0
        while i < n {
            if beams[i].isEmpty {
                i += 1
                continue
            }
            let currentStates = beams[i]

            let maxL = min(Self.maxWordLength, n - i)
            var l = maxL
            while 1 <= l {
                let subStr = String(chars[i..<(i + l)])
                let exactEntries = dictionary.lookupExact(reading: subStr)

                for entry in exactEntries {
                    let emissionLogProb = dictionary.logReadingGivenSurface(reading: subStr, surface: entry.surface)
                    if emissionLogProb.isFinite != true {
                        continue
                    }
                    let emitted = entry.surface + commaSuffix(after: entry.surface)
                    for st in currentStates {
                        let bigramLogProb = dictionary.logBigram(from: st.lastWord, to: entry.surface)
                        let stepScore = emissionLogProb + (bigramWeight * bigramLogProb)
                        push(i + l, BeamState(
                            score: st.score + stepScore,
                            text: st.text + emitted,
                            lastWord: entry.surface,
                            trace: st.trace + [TraceSegment(
                                kanaRange: subStr, emitted: emitted,
                                kind: "exact", stepScore: stepScore
                            )]
                        ))
                    }
                }

                if Self.minFuzzyLength <= l && exactEntries.isEmpty {
                    let fuzzyList = dictionary.lookupFuzzyPhonetic(reading: subStr, maxPhoneticDist: Self.maxFuzzyDistance)
                    for item in fuzzyList {
                        let entry = item.entry
                        let emissionLogProb = dictionary.logReadingGivenSurface(reading: entry.reading, surface: entry.surface)
                        if emissionLogProb.isFinite != true {
                            continue
                        }
                        let emitted = entry.surface + commaSuffix(after: entry.surface)
                        let fuzzyCost = Self.fuzzyDistanceCost * item.dist
                        for st in currentStates {
                            let bigramLogProb = dictionary.logBigram(from: st.lastWord, to: entry.surface)
                            let stepScore = emissionLogProb + (bigramWeight * bigramLogProb) + fuzzyCost
                            push(i + l, BeamState(
                                score: st.score + stepScore,
                                text: st.text + emitted,
                                lastWord: entry.surface,
                                trace: st.trace + [TraceSegment(
                                    kanaRange: subStr, emitted: emitted,
                                    kind: "fuzzy(dist \(String(format: "%.2f", item.dist)))",
                                    stepScore: stepScore
                                )]
                            ))
                        }
                    }
                }

                l -= 1
            }

            let singleChar = String(chars[i])
            for st in currentStates {
                push(i + 1, BeamState(
                    score: st.score + Self.unknownCharLogProb,
                    text: st.text + singleChar,
                    lastWord: singleChar,
                    trace: st.trace + [TraceSegment(
                        kanaRange: singleChar, emitted: singleChar,
                        kind: "pass-through", stepScore: Self.unknownCharLogProb
                    )]
                ))
            }

            i += 1
        }

        let suffix = sentenceFinalSuffix()
        if let best = beams[n].first {
            lastTrace = best.trace
        }
        return beams[n].map { st in
            var restored = st.text
            if let last = restored.last {
                if punctuationCharacters.contains(last) {
                    restored.removeLast()
                }
            }
            return Candidate(text: restored + suffix, score: st.score)
        }
    }

    /// Converts the acoustic model's kana output to kanji-kana text.
    public func decode(kanaText: String) -> String {
        let candidates = decodeNBest(kanaText: kanaText, beamWidth: Self.defaultBeamWidth)
        switch candidates.first {
        case .some(let best):
            return best.text
        case .none:
            return ""
        }
    }
}
