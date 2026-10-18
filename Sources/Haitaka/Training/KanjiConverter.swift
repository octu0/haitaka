import Foundation

/// Tokenizes Japanese text and converts it into pronunciation kana (hiragana) used as training targets.
public struct KanjiConverter: Sendable {
    /// Pronouncing dictionary for English words. When nil, unknown Latin-script words get an empty reading.
    public let english: EnglishPronunciations?

    public init(english: EnglishPronunciations? = nil) {
        self.english = english
    }

    /// Surface form and reading of one morpheme.
    public struct Token: Sendable {
        public let surface: String
        public let reading: String

        public init(surface: String, reading: String) {
            self.surface = surface
            self.reading = reading
        }
    }

    static let digitKana = ["ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]

    /// Digits of a numeric string (ASCII or full-width) as 0-9 values, or nil if any non-digit is present.
    static func digitValues(_ text: String) -> [Int]? {
        var values: [Int] = []
        for c in text.unicodeScalars {
            switch c.value {
            case 0x30...0x39:
                values.append(Int(c.value) - 0x30)
            case 0xFF10...0xFF19:
                values.append(Int(c.value) - 0xFF10)
            default:
                return nil
            }
        }
        if values.isEmpty {
            return nil
        }
        return values
    }

    /// Reading of a 4-digit group (0-9999), including sound changes (さんびゃく, ろっぴゃく, はっせん, ...).
    static func fourDigitReading(_ value: Int) -> String {
        var result = ""
        let thousands = value / 1000
        let hundreds = (value / 100) % 10
        let tens = (value / 10) % 10
        let ones = value % 10
        switch thousands {
        case 0:
            break
        case 1:
            result += "せん"
        case 3:
            result += "さんぜん"
        case 8:
            result += "はっせん"
        default:
            result += digitKana[thousands] + "せん"
        }
        switch hundreds {
        case 0:
            break
        case 1:
            result += "ひゃく"
        case 3:
            result += "さんびゃく"
        case 6:
            result += "ろっぴゃく"
        case 8:
            result += "はっぴゃく"
        default:
            result += digitKana[hundreds] + "ひゃく"
        }
        switch tens {
        case 0:
            break
        case 1:
            result += "じゅう"
        default:
            result += digitKana[tens] + "じゅう"
        }
        if 0 < ones {
            result += digitKana[ones]
        }
        return result
    }

    /// Kana reading of a digit string, read by place value in 4-digit groups (万, 億, ...).
    /// Strings with a leading zero (phone numbers etc.) or 17+ digits are read digit by digit.
    static func numberReading(_ text: String) -> String? {
        guard let values = digitValues(text) else {
            return nil
        }
        let digitWise = { () -> String in
            var r = ""
            for v in values {
                r += digitKana[v]
            }
            return r
        }
        if values[0] == 0 || 16 < values.count {
            if values.count == 1 {
                return "ぜろ"
            }
            return digitWise()
        }
        var number = 0
        for v in values {
            number = number * 10 + v
        }
        let groupUnits = ["", "まん", "おく", "ちょう"]
        var groups: [Int] = []
        var rest = number
        while 0 < rest {
            groups.append(rest % 10000)
            rest /= 10000
        }
        var result = ""
        var gi = groups.count - 1
        while 0 <= gi {
            let g = groups[gi]
            if 0 < g {
                // 1万 and 1億 are read 「いちまん」/「いちおく」 (unlike 千 or 百, the いち is not dropped).
                if g == 1 && 0 < gi {
                    result += "いち"
                } else {
                    result += fourDigitReading(g)
                }
                result += groupUnits[gi]
            }
            gi -= 1
        }
        return result
    }

    /// Whether the scalar is a long-vowel mark or a stretch mark (wave dash, tildes, half-width ｰ, bars, dashes).
    static func isProlongedMark(_ val: UInt32) -> Bool {
        switch val {
        case 0x301C, // WAVE DASH (〜)
             0xFF5E, // FULLWIDTH TILDE (～)
             0x3030, // WAVY DASH (〰)
             0x223C, // TILDE OPERATOR (∼)
             0xFF70, // HALFWIDTH KATAKANA-HIRAGANA PROLONGED SOUND MARK (ｰ)
             0x2015, // HORIZONTAL BAR (―)
             0x2014, // EM DASH (—)
             0x2013, // EN DASH (–)
             0x2500, // BOX DRAWINGS LIGHT HORIZONTAL (─)
             0xFF0D: // FULLWIDTH HYPHEN-MINUS (－)
            return true
        default:
            return false
        }
    }

    /// Normalizes stretch marks such as the wave dash to the long-vowel mark 「ー」.
    static func normalizeProlongedMarks(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            if isProlongedMark(scalar.value) {
                result.append("ー")
            } else {
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    /// Removes bracketed laugh annotations containing 「笑」 (笑, 爆笑, 冷笑); they describe laughter and are not spoken as 「わらい」.
    /// Only contents up to 6 characters are removed, so long bracketed quotes remain.
    /// A bare 「w」 is kept because it is spoken as 「だぶりゅー」.
    static func removeLaughAnnotations(_ text: String) -> String {
        let chars = Array(text)
        var out: [Character] = []
        out.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == "(" || c == "（" {
                var j = i + 1
                var hasLaugh = false
                while j < chars.count && j - i <= 7 && chars[j] != ")" && chars[j] != "）" {
                    if chars[j] == "笑" {
                        hasLaugh = true
                    }
                    j += 1
                }
                if hasLaugh && j < chars.count && (chars[j] == ")" || chars[j] == "）") {
                    i = j + 1
                    continue
                }
            }
            out.append(c)
            i += 1
        }
        return String(out)
    }

    /// Whether the span is a hyphen, dash, or stretch mark.
    static func isHyphenSpan(_ surface: String) -> Bool {
        if surface.count != 1 {
            return false
        }
        guard let val = surface.unicodeScalars.first?.value else {
            return false
        }
        switch val {
        case 0x2D,   // HYPHEN-MINUS (-)
             0xFF0D, // FULLWIDTH HYPHEN-MINUS (－)
             0x2010, // HYPHEN (‐)
             0x2011, // NON-BREAKING HYPHEN (‑)
             0x2012, // FIGURE DASH (‒)
             0x2013, // EN DASH (–)
             0x2014, // EM DASH (—)
             0x2015, // HORIZONTAL BAR (―)
             0x30FC: // PROLONGED SOUND MARK (ー)
            return true
        default:
            return false
        }
    }

    /// Whether the token is Latin-script (ASCII or full-width letters, or a hyphenated English compound).
    static func isAlphabetToken(_ text: String) -> Bool {
        if text.isEmpty {
            return false
        }
        var letterCount = 0
        for scalar in text.unicodeScalars {
            let v = scalar.value
            switch v {
            case 0x41...0x5A, 0x61...0x7A, 0xFF21...0xFF3A, 0xFF41...0xFF5A:
                letterCount += 1
            case 0x2D, 0xFF0D, 0x2010...0x2015:
                ()
            default:
                return false
            }
        }
        return 0 < letterCount
    }

    /// Reading of a Latin-script token.
    ///
    /// `SeedVocabulary` takes priority; all-caps acronyms are spelled letter by letter.
    /// Unknown English words get an empty reading rather than a romaji-rule reading (e.g. 「いぷほね」 for iPhone).
    func alphabetReading(_ surface: String) -> String? {
        // Lowercase single-letter a / i are the English article and pronoun; read them from the dictionary, not as letter names (えー).
        if surface.count == 1, surface == surface.lowercased(), let dictionary = english,
           let word = dictionary.reading(of: surface) {
            return word
        }
        if let reading = SeedVocabulary.readingForAlphabetSurface(surface) {
            return reading
        }

        if Self.isAlphabetToken(surface) {
            // All-caps acronyms ("GPU", "API", "PC", "USB") are spelled letter by letter.
            var isAllUpper = true
            var letterCount = 0
            for scalar in surface.unicodeScalars {
                let v = scalar.value
                switch v {
                case 0x41...0x5A, 0xFF21...0xFF3A:
                    letterCount += 1
                case 0x2D, 0xFF0D, 0x2010...0x2015:
                    ()
                default:
                    isAllUpper = false
                    break
                }
                if isAllUpper != true {
                    break
                }
            }

            // All-caps words of 5+ letters are usually pronounced as words (ZENSHO), not spelled.
            // Their reading is unknown, so fall through to the dictionary instead of spelling them.
            if isAllUpper && 0 < letterCount && letterCount <= 4 {
                var spelled = ""
                for scalar in surface.unicodeScalars {
                    let letter = String(scalar)
                    if let r = SeedVocabulary.readingForLetter(letter) {
                        spelled.append(r)
                    }
                }
                if spelled.isEmpty != true {
                    return spelled
                }
            }

            // Look up the pronouncing dictionary; empty if not found, to avoid a broken romaji reading.
            return englishReading(surface)
        }

        // Tokens mixing letters and punctuation (dream.See, Baby,) are looked up per run of letters.
        // Words missing from the dictionary get an empty reading instead of the tokenizer's romaji reading (でれあむせー).
        if Self.containsLatinLetter(surface) {
            return englishReading(surface)
        }

        return nil
    }

    /// Looks up each run of Latin letters in the pronouncing dictionary and concatenates the readings; missing words contribute nothing.
    func englishReading(_ surface: String) -> String {
        guard let dictionary = english else {
            return ""
        }
        var reading = ""
        var word = ""
        for scalar in surface.unicodeScalars {
            switch scalar.value {
            case 0x41...0x5A, 0x61...0x7A:
                word.unicodeScalars.append(scalar)
            case 0xFF21...0xFF3A, 0xFF41...0xFF5A:
                if let half = UnicodeScalar(scalar.value - 0xFEE0) {
                    word.unicodeScalars.append(half)
                }
            case 0x27:
                word.unicodeScalars.append(scalar)
            default:
                if word.isEmpty != true {
                    reading += dictionary.reading(of: word) ?? ""
                    word = ""
                }
            }
        }
        if word.isEmpty != true {
            reading += dictionary.reading(of: word) ?? ""
        }
        return reading
    }

    static func containsLatinLetter(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x41...0x5A, 0x61...0x7A, 0xFF21...0xFF3A, 0xFF41...0xFF5A:
                return true
            default:
                continue
            }
        }
        return false
    }

    /// Splits text into morphemes, returning surface forms and readings together.
    ///
    /// Readings come from the tokenizer, which picks them in context; a static per-kanji table
    /// cannot decide whether 「人」 is ひと, にん, or じん.
    public func tokenize(_ text: String) -> [Token] {
        if text.isEmpty {
            return []
        }

        // Normalize stretch marks (〜 etc.), remove unspoken laugh annotations, and spell out 「痛ーい」-style stretches.
        let cleanText = stretchedOkurigana(Self.removeLaughAnnotations(Self.normalizeProlongedMarks(text)))

        let loc = Locale(identifier: "ja_JP") as CFLocale
        let nsText = cleanText as NSString
        let tokenizer = CFStringTokenizerCreate(
            nil,
            cleanText as CFString,
            CFRangeMake(0, nsText.length),
            kCFStringTokenizerUnitWordBoundary,
            loc
        )

        struct RawTokenSpan {
            let surface: String
            let range: CFRange
            let latinAttr: NSString?
        }

        var rawSpans: [RawTokenSpan] = []
        while CFStringTokenizerAdvanceToNextToken(tokenizer) != [] {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let surface = nsText.substring(with: NSRange(location: range.location, length: range.length))
            let attr = CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, kCFStringTokenizerAttributeLatinTranscription) as? NSString
            rawSpans.append(RawTokenSpan(surface: surface, range: range, latinAttr: attr))
        }

        // Join hyphenated English words split by the tokenizer ("Wi" + "-" + "Fi" -> "Wi-Fi").
        var mergedSpans: [RawTokenSpan] = []
        var spanIdx = 0
        while spanIdx < rawSpans.count {
            var currentSpan = rawSpans[spanIdx]
            spanIdx += 1

            // Join when a hyphen and then a Latin token follow with no whitespace in between.
            while spanIdx + 1 < rawSpans.count {
                let hyphenSpan = rawSpans[spanIdx]
                let nextSpan = rawSpans[spanIdx + 1]
                let isContiguousHyphen = Self.isHyphenSpan(hyphenSpan.surface) &&
                    currentSpan.range.location + currentSpan.range.length == hyphenSpan.range.location &&
                    hyphenSpan.range.location + hyphenSpan.range.length == nextSpan.range.location
                if isContiguousHyphen && Self.isAlphabetToken(currentSpan.surface) && Self.isAlphabetToken(nextSpan.surface) {
                    let combinedSurface = "\(currentSpan.surface)-\(nextSpan.surface)"
                    let combinedRange = CFRangeMake(
                        currentSpan.range.location,
                        currentSpan.range.length + hyphenSpan.range.length + nextSpan.range.length
                    )
                    currentSpan = RawTokenSpan(surface: combinedSurface, range: combinedRange, latinAttr: nil)
                    spanIdx += 2
                } else {
                    break
                }
            }
            mergedSpans.append(currentSpan)
        }

        // The tokenizer skips punctuation, so the marks are picked up from the gaps between tokens.
        func appendPunctuation(from start: CFIndex, to end: CFIndex) {
            if end <= start {
                return
            }
            for mark in kanaOnly(nsText.substring(with: NSRange(location: start, length: end - start))) {
                if Self.punctuation.contains(mark) {
                    tokens.append(Token(surface: String(mark), reading: String(mark)))
                }
            }
        }

        var tokens: [Token] = []
        var consumedEnd: CFIndex = 0
        var mIdx = 0
        while mIdx < mergedSpans.count {
            let span = mergedSpans[mIdx]
            mIdx += 1
            appendPunctuation(from: consumedEnd, to: span.range.location)
            consumedEnd = span.range.location + span.range.length
            let surface = span.surface

            // Punctuation tokens read as themselves; the tokenizer's Latin transcription would drop them.
            let marks = kanaOnly(surface)
            if marks.isEmpty != true && marks.allSatisfy({ Self.punctuation.contains($0) }) {
                tokens.append(Token(surface: surface, reading: marks))
                continue
            }

            // Numbers get a place-value kana reading generated directly.
            if let numReading = Self.numberReading(surface) {
                tokens.append(Token(surface: surface, reading: Self.pronunciation(surface: surface, reading: numReading)))
                continue
            }

            // Latin-script tokens (Wi-Fi, iPhone, ...).
            if let alphaReading = alphabetReading(surface) {
                tokens.append(Token(surface: surface, reading: kanaOnly(alphaReading)))
                continue
            }

            var reading = ""
            switch span.latinAttr {
            case .some(let latin):
                // Decompose (NFD) so macrons become the combining U+0304, then replace it with 「ー」 before CFStringTransform.
                let decomposed = (latin as String).decomposedStringWithCanonicalMapping
                let withProlongedMark = decomposed.replacingOccurrences(of: "\u{0304}", with: "ー")
                let ms = NSMutableString(string: withProlongedMark)
                CFStringTransform(ms as CFMutableString, nil, kCFStringTransformLatinHiragana, false)
                reading = normalizeKana(ms as String)
            case .none:
                reading = normalizeKana(surface)
            }

            tokens.append(Token(surface: surface, reading: kanaOnly(Self.pronunciation(surface: surface, reading: reading))))
        }

        appendPunctuation(from: consumedEnd, to: nsText.length)
        return Self.applyReadingOverrides(tokens)
    }

    /// Single-morpheme surfaces whose tokenizer reading is replaced by the spoken reading. Values are orthographic readings passed through `pronunciation`.
    static let readingOverrides: [String: String] = [
        "私": "わたし",
        "日本": "にほん",
        "明日": "あした",
        "皆": "みんな",
        "言う": "ゆー",
        "いう": "ゆー",
        "良い": "いい",
        "得る": "える",
        "近々": "ちかぢか",
        "白髪": "しらが",
        "玩具": "おもちゃ",
        "獰猛": "どうもう",
        "寝惚け": "ねぼけ",
        "貴女": "あなた",
        "蛙": "かえる",
    ]

    /// Overrides applied only when the tokenizer returns a specific reading: (surface, pronunciation-kana reading) -> reading.
    /// 「下」 varies by context (した, か, げ), so only the reading もと, rare in speech, is corrected.
    static let readingSpecificOverrides: [String: [String: String]] = [
        "下": ["もと": "した"],
        "米": ["べー": "こめ"],
    ]

    /// Words whose reading changes only before a particle (他の -> ほかの, 数は -> かずは).
    static let beforeParticleOverrides: [String: String] = [
        "他": "ほか",
        "数": "かず",
    ]
    static let particles: Set<String> = ["の", "は", "が", "を", "も", "に", "へ", "と", "で", "から", "まで", "より"]

    /// Suffixes that undergo rendaku (sequential voicing) after a noun (ビル作り -> づくり, 多摩川 -> がわ).
    /// Applied only when the previous morpheme ends in kanji or katakana, so 「この通り」 and 「深い川」 are unchanged.
    static let rendakuSuffixes: [String: String] = [
        "川": "がわ", "作り": "づくり", "造り": "づくり", "好き": "ずき", "通り": "どおり",
        "時計": "どけい", "頃": "ごろ", "不足": "ぶそく", "気味": "ぎみ", "部屋": "べや", "会社": "がいしゃ",
    ]

    /// Alternative spoken readings not chosen in `readingOverrides`.
    /// The training target can carry only one reading (whether the speaker said 「あたし」 cannot be known from the text),
    /// so these are registered only in the stage-2 dictionary, mapping any reading the acoustic model emits back to the same surface.
    public static let readingVariants: [String: [String]] = [
        "私": ["あたし", "わたくし"],
        "皆": ["みな"],
    ]

    /// Multi-morpheme phrases matched on concatenated surfaces and merged into one morpheme.
    /// Needed because the tokenizer splits words like 「お/母/さん」 and 「木曜/日」 and reads each piece as a standalone word.
    static let phraseOverrides: [String: String] = [
        "お母さん": "おかあさん",
        "お父さん": "おとうさん",
        "お兄さん": "おにいさん",
        "お姉さん": "おねえさん",
        "皆さん": "みなさん",
        "皆様": "みなさま",
        "兄さん": "にいさん",
        "姉さん": "ねえさん",
        "兄ちゃん": "にいちゃん",
        "姉ちゃん": "ねえちゃん",
        "父さん": "とうさん",
        "母さん": "かあさん",
        "爺ちゃん": "じいちゃん",
        "婆ちゃん": "ばあちゃん",
        "一昨日": "おととい",
        "一昨年": "おととし",
        "二日": "ふつか",
        "四人": "よにん",
        "世界中": "せかいじゅう",
        "日本中": "にほんじゅう",
        "一日中": "いちにちじゅう",
    ]

    static let maxPhraseTokens = 3

    /// Words after which 「君」 is the pronoun きみ rather than the name suffix くん.
    static let kimiPredecessors: Set<String> = [
        "は", "が", "を", "に", "の", "と", "も", "で", "へ", "や", "から", "まで", "でも", "だから", "そして",
        "また", "まだ", "もう", "ずっと", "ねえ", "ねぇ", "ああ", "いつも", "だけ", "きっと", "ただ", "なぜ", "もし",
    ]

    /// Base readings of counters whose tokenizer reading varies with the preceding number (e.g. 「本」 as ぽん), normalized before sandhi.
    static let counterBaseReadings: [String: String] = [
        "本": "ほん", "杯": "はい", "匹": "ひき", "分": "ふん", "歩": "ほ",
        "発": "はつ", "泊": "はく", "品": "ひん", "敗": "はい", "拍": "はく",
    ]

    /// Counters that become voiced (ぼん) after numbers ending in ん (さん, せん, まん, なん); other counters become semi-voiced (さんぷん).
    static let voicedAfterNasal: Set<String> = ["本", "杯", "匹"]

    /// Number + counter combinations with irregular readings (days, months, hours, people, つ). Values are orthographic readings passed through `pronunciation`.
    /// Days not in the table are read "N にち" (the tokenizer reads 11日 as 「じゅういち か」).
    static let numeralCounterReadings: [String: [Int: String]] = [
        "日": [1: "いちにち", 2: "ふつか", 3: "みっか", 4: "よっか", 5: "いつか", 6: "むいか", 7: "なのか", 8: "ようか",
               9: "ここのか", 10: "とおか", 14: "じゅうよっか", 20: "はつか", 24: "にじゅうよっか"],
        "月": [4: "しがつ", 7: "しちがつ", 9: "くがつ"],
        "時": [4: "よじ", 7: "しちじ", 9: "くじ"],
        "人": [1: "ひとり", 2: "ふたり", 4: "よにん"],
        "つ": [1: "ひとつ", 2: "ふたつ", 3: "みっつ", 4: "よっつ", 5: "いつつ", 6: "むっつ", 7: "ななつ", 8: "やっつ",
               9: "ここのつ", 10: "とお"],
    ]

    static let kanjiNumeralValues: [String: Int] = [
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十四": 14, "二十": 20, "二十四": 24,
    ]

    /// Numeric value of a numeral surface: Arabic digits and kanji numerals up to 24.
    static func numeralValue(_ surface: String) -> Int? {
        if let values = digitValues(surface) {
            if 4 < values.count {
                return nil
            }
            var v = 0
            for d in values {
                v = v * 10 + d
            }
            return v
        }
        return kanjiNumeralValues[surface]
    }

    /// Irregular reading of a numeral + counter pair, or nil. Tokens the tokenizer merged (e.g. 「1人」 read れん) are split by `splitNumeralCounter` first.
    static func numeralCounterReading(numeral: String, numeralReading: String, counter: String) -> String? {
        guard let table = numeralCounterReadings[counter], let value = numeralValue(numeral) else {
            return nil
        }
        if let reading = table[value] {
            return reading
        }
        if counter == "日" {
            return numeralReading + "にち"
        }
        return nil
    }

    /// Splits a surface whose last character is a counter in the table and whose prefix is a numeral (「1人」, 「10日」).
    static func splitNumeralCounter(_ surface: String) -> (String, String)? {
        guard 1 < surface.count, let last = surface.last, numeralCounterReadings[String(last)] != nil else {
            return nil
        }
        let numeral = String(surface.dropLast())
        if numeralValue(numeral) == nil {
            return nil
        }
        return (numeral, String(last))
    }

    /// Whether the surface is a numeral (Arabic digits, kanji numerals, 何, 数, 幾).
    static func isNumeralSurface(_ surface: String) -> Bool {
        if surface.isEmpty {
            return false
        }
        for scalar in surface.unicodeScalars {
            switch scalar.value {
            case 0x30...0x39, 0xFF10...0xFF19:
                continue
            default:
                break
            }
            if "〇一二三四五六七八九十百千万億兆何数幾".unicodeScalars.contains(scalar) != true {
                return false
            }
        }
        return true
    }

    static func isKanji(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x4E00...0x9FFF, 0x3400...0x4DBF:
            return true
        default:
            return false
        }
    }

    static func isKatakana(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x30A1...0x30FA:
            return true
        default:
            return false
        }
    }

    /// Whether the surface starts with kanji or katakana.
    static func startsWithKanjiOrKatakana(_ surface: String) -> Bool {
        guard let first = surface.unicodeScalars.first else {
            return false
        }
        return isKanji(first) || isKatakana(first)
    }

    /// Whether the surface ends with kanji or katakana (excludes verbs and adjectives ending in kana inflections).
    static func endsWithKanjiOrKatakana(_ surface: String) -> Bool {
        guard let last = surface.unicodeScalars.last else {
            return false
        }
        return isKanji(last) || isKatakana(last)
    }

    /// Whether the reading starts with a k/s/t/h/p-row kana, the consonants that geminate a preceding numeral.
    static func startsWithVoicelessObstruent(_ reading: String) -> Bool {
        guard let first = reading.first else {
            return false
        }
        return "かきくけこさしすせそたちつてとはひふへほぱぴぷぺぽ".contains(first)
    }

    /// Maps h-row kana to p-row or b-row.
    static func replacingHRow(_ reading: String, with row: String) -> String {
        guard let first = reading.first, let idx = "はひふへほ".firstIndex(of: first) else {
            return reading
        }
        let offset = "はひふへほ".distance(from: "はひふへほ".startIndex, to: idx)
        let rowChars = Array(row)
        return String(rowChars[offset]) + String(reading.dropFirst())
    }

    /// Numeral + counter sandhi: gemination of the numeral ending (いち -> いっ) and (semi-)voicing of the counter onset (ほん -> ぽん, ぼん).
    static func counterSandhi(numeral: Token, counter: Token) -> (String, String)? {
        if isNumeralSurface(numeral.surface) != true || startsWithKanjiOrKatakana(counter.surface) != true {
            if counter.surface.hasPrefix("か月") != true {
                return nil
            }
        }
        var base = counter.reading
        if let known = counterBaseReadings[counter.surface] {
            base = known
        }
        // Katakana loanword units do not trigger gemination (いちへくたーる).
        if let first = counter.surface.unicodeScalars.first, isKatakana(first), let head = base.first, "はひふへほ".contains(head) {
            return nil
        }
        let n = numeral.reading
        let geminating: [(String, String, Bool)] = [
            ("いち", "いっ", true), ("はち", "はっ", true), ("じゅー", "じゅっ", true),
            ("ろく", "ろっ", false), ("ひゃく", "ひゃっ", false),
        ]
        for (tail, geminated, allRows) in geminating {
            if n.hasSuffix(tail) != true || startsWithVoicelessObstruent(base) != true {
                continue
            }
            let isKOrH = "かきくけこはひふへほ".contains(base.first!)
            if allRows != true && isKOrH != true {
                return nil
            }
            return (String(n.dropLast(tail.count)) + geminated, replacingHRow(base, with: "ぱぴぷぺぽ"))
        }
        if counterBaseReadings[counter.surface] == nil {
            return nil
        }
        for tail in ["さん", "せん", "まん", "なん"] {
            if n.hasSuffix(tail) != true {
                continue
            }
            if voicedAfterNasal.contains(counter.surface) {
                return (n, replacingHRow(base, with: "ばびぶべぼ"))
            }
            return (n, replacingHRow(base, with: "ぱぴぷぺぽ"))
        }
        if n.hasSuffix("よん") && voicedAfterNasal.contains(counter.surface) != true {
            return (n, replacingHRow(base, with: "ぱぴぷぺぽ"))
        }
        return (n, base)
    }

    /// Replaces tokenizer readings that do not match how the words are actually spoken.
    ///
    /// The tokenizer returns 「私/わたくし」, 「皆/みな」, 「日本/にっぽん」, splits compounds like 「お/母/さん」 and 「木曜/日」
    /// with per-piece readings, and ignores counter sandhi (いっぽん, さんぼん).
    /// Training targets must match the actual sounds, so they are normalized here.
    static func applyReadingOverrides(_ tokens: [Token]) -> [Token] {
        var result: [Token] = []
        var i = 0
        while i < tokens.count {
            // 1. Multi-morpheme phrases (longest first)
            var span = min(maxPhraseTokens, tokens.count - i)
            var merged = false
            while 1 < span {
                var surface = ""
                var k = 0
                while k < span {
                    surface += tokens[i + k].surface
                    k += 1
                }
                if let reading = phraseOverrides[surface] {
                    result.append(Token(surface: surface, reading: pronunciation(surface: surface, reading: reading)))
                    i += span
                    merged = true
                    break
                }
                span -= 1
            }
            if merged {
                continue
            }
            var token = tokens[i]
            // 2. Single-morpheme overrides
            if let reading = readingOverrides[token.surface] {
                token = Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading))
            }
            if let reading = readingSpecificOverrides[token.surface]?[token.reading] {
                token = Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading))
            }
            if let reading = beforeParticleOverrides[token.surface], i + 1 < tokens.count, particles.contains(tokens[i + 1].surface) {
                token = Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading))
            }
            if let reading = rendakuSuffixes[token.surface], let prev = result.last, endsWithKanjiOrKatakana(prev.surface) {
                token = Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading))
            }
            // 君: くん after a name (たかし君); the pronoun きみ at the start or after particles and conjunctions.
            if token.surface == "君" && token.reading == "くん" {
                var isPronoun = true
                if let prev = result.last, prev.reading.isEmpty != true, kimiPredecessors.contains(prev.surface) != true {
                    isPronoun = false
                }
                if isPronoun {
                    token = Token(surface: token.surface, reading: "きみ")
                }
            }
            // 3. Weekdays: 「木曜/もくよう」 + 「日/ひ」 -> もくようび
            if i + 1 < tokens.count && token.surface.hasSuffix("曜") && tokens[i + 1].surface == "日" {
                result.append(Token(surface: token.surface + "日", reading: token.reading + "び"))
                i += 2
                continue
            }
            // 4. Counters whose reading depends on the number (ついたち, しがつ, よじ, ひとり, みっつ)
            if i + 1 < tokens.count, isNumeralSurface(token.surface),
               let reading = numeralCounterReading(numeral: token.surface, numeralReading: token.reading, counter: tokens[i + 1].surface) {
                let surface = token.surface + tokens[i + 1].surface
                result.append(Token(surface: surface, reading: pronunciation(surface: surface, reading: reading)))
                i += 2
                continue
            }
            if let (numeral, counter) = splitNumeralCounter(token.surface),
               let reading = numeralCounterReading(numeral: numeral, numeralReading: pronunciation(surface: numeral, reading: numberReading(numeral) ?? ""), counter: counter) {
                result.append(Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading)))
                i += 1
                continue
            }
            // 5. Numeral + counter sandhi
            if i + 1 < tokens.count, let (numeral, counter) = counterSandhi(numeral: token, counter: tokens[i + 1]) {
                result.append(Token(surface: token.surface, reading: numeral))
                result.append(Token(surface: tokens[i + 1].surface, reading: counter))
                i += 2
                continue
            }
            // 6. Country or group name + 人 -> じん (excluding numerals, e.g. 「三人/さんにん」)
            if token.surface == "人" && token.reading == "にん", let prev = result.last,
               startsWithKanjiOrKatakana(prev.surface) && isNumeralSurface(prev.surface) != true {
                token = Token(surface: token.surface, reading: "じん")
            }
            // 7. 何 is なに before particles and verbs (何が, 何を, 何か, 何して); it stays なん before counters and in 「何で」.
            if token.surface == "何" && i + 1 < tokens.count, let next = tokens[i + 1].surface.first {
                if "がをもよしやか".contains(next) {
                    token = Token(surface: token.surface, reading: "なに")
                }
            }
            result.append(token)
            i += 1
        }
        return result
    }

    /// Vowel (あ/い/う/え/お) of a kana, or nil for ん, っ, ー, and symbols.
    static func vowel(of kana: Character) -> Character? {
        switch kana {
        case "あ", "か", "が", "さ", "ざ", "た", "だ", "な", "は", "ば", "ぱ", "ま", "や", "ら", "わ", "ぁ", "ゃ", "ゎ":
            return "あ"
        case "い", "き", "ぎ", "し", "じ", "ち", "ぢ", "に", "ひ", "び", "ぴ", "み", "り", "ゐ", "ぃ":
            return "い"
        case "う", "く", "ぐ", "す", "ず", "つ", "づ", "ぬ", "ふ", "ぶ", "ぷ", "む", "ゆ", "る", "ぅ", "ゅ", "ゔ":
            return "う"
        case "え", "け", "げ", "せ", "ぜ", "て", "で", "ね", "へ", "べ", "ぺ", "め", "れ", "ゑ", "ぇ":
            return "え"
        case "お", "こ", "ご", "そ", "ぞ", "と", "ど", "の", "ほ", "ぼ", "ぽ", "も", "よ", "ろ", "を", "ぉ", "ょ":
            return "お"
        default:
            return nil
        }
    }

    /// Verbs whose final 「う」 is kept rather than lengthened (dictionary form ending in an o- or u-row kana + う).
    /// Volitional forms (行こう, 見よう) and kana words such as 「そう」, 「もう」, 「ありがとう」 are long vowels, so only listed verbs are kept.
    static let uEndingVerbs: Set<String> = [
        "思う", "追う", "問う", "酔う", "沿う", "食う", "吸う", "縫う", "乞う", "負う", "覆う", "請う",
        "装う", "添う", "集う", "揃う", "争う", "救う", "狂う", "通う", "見舞う", "住まう", "買う", "会う",
    ]

    /// Normalizes an orthographic reading to pronunciation kana.
    ///
    /// Training targets must correspond to sounds. In orthography the same sound [oː] is written
    /// 「おう」, 「おお」, or 「ー」, and the particles 「は」 and 「を」 are pronounced わ and お.
    /// The stage-2 kana-to-kanji dictionary is built from the same readings, so the pipeline stays consistent.
    ///   - Particles 「は」, 「へ」 (standalone tokens) -> わ, え; 「を」 -> お
    ///   - ぢ, づ -> じ, ず
    ///   - The second kana of a same-vowel pair (おう, おお, えい, ええ, ああ, いい, うう) -> ー,
    ///     except the final う of verbs in `uEndingVerbs` (思う, 追う, 食う).
    static func pronunciation(surface: String, reading: String) -> String {
        switch surface {
        case "は":
            return "わ"
        case "へ":
            return "え"
        default:
            break
        }
        let chars = Array(reading)
        var result: [Character] = []
        result.reserveCapacity(chars.count)
        let keepsFinalU = Self.uEndingVerbs.contains(surface)
        var i = 0
        while i < chars.count {
            var c = chars[i]
            switch c {
            case "を":
                c = "お"
            case "ぢ":
                c = "じ"
            case "づ":
                c = "ず"
            default:
                break
            }
            if 0 < i, let prevVowel = Self.vowel(of: result[result.count - 1]) {
                let isLast = (i == chars.count - 1)
                var lengthens = false
                switch c {
                case "う":
                    lengthens = (prevVowel == "お" || prevVowel == "う") && (isLast && keepsFinalU) != true
                case "い":
                    lengthens = (prevVowel == "え" || prevVowel == "い")
                case "え":
                    lengthens = (prevVowel == "え")
                case "お":
                    lengthens = (prevVowel == "お")
                case "あ":
                    lengthens = (prevVowel == "あ")
                default:
                    break
                }
                if lengthens {
                    c = "ー"
                }
            }
            result.append(c)
            i += 1
        }
        return String(result)
    }

    /// The punctuation marks kept in kana text: 、。？！. They mark pauses and intonation, so the acoustic
    /// model learns them as targets and the kana n-gram LM learns where they go. Scoring strips them
    /// again (`stripPunctuation` in train), so kana CER stays comparable.
    public static let punctuation: Set<Character> = ["、", "。", "？", "！"]

    /// Keeps hiragana, the long-vowel mark and the four punctuation marks; ASCII and full-width
    /// variants of the marks are folded into 、。？！.
    public func kanaOnly(_ text: String) -> String {
        // Normalize to precomposed form, then select scalar by scalar. Checking by grapheme cluster
        // would let kana with combining marks (e.g. 「を」 + dakuten) through as one character and grow the vocabulary without bound.
        var result = ""
        for scalar in text.precomposedStringWithCanonicalMapping.unicodeScalars {
            let val = scalar.value
            switch val {
            case 0x3041...0x3096, 0x30FC:
                result.unicodeScalars.append(scalar)
            case _ where Self.isProlongedMark(val):
                result.append("ー")
            case 0x3001, 0xFF0C, 0x2C:
                result.append("、")
            case 0x3002, 0xFF0E, 0x2E:
                result.append("。")
            case 0xFF1F, 0x3F:
                result.append("？")
            case 0xFF01, 0x21:
                result.append("！")
            default:
                break
            }
        }
        return result
    }

    /// Collapses runs of punctuation to their first mark, drops a leading mark, and collapses runs of 「ー」
    /// (「ありがとうー」 reads 「ありがとー」; CTC cannot emit a repeated 「ー」 without a blank between them anyway).
    static func tidyKana(_ text: String) -> String {
        var result = ""
        var lastWasPunctuation = true
        var lastWasLong = false
        for c in text {
            if punctuation.contains(c) {
                if lastWasPunctuation != true {
                    result.append(c)
                }
                lastWasPunctuation = true
                lastWasLong = false
            } else {
                if c != "ー" || lastWasLong != true {
                    result.append(c)
                }
                lastWasPunctuation = false
                lastWasLong = c == "ー"
            }
        }
        return result
    }

    /// Rewrites 「漢字ー送り仮名」 (痛ーい, 凄ーい) as kana with the stretch after the stem (いたーい). Left as is, the
    /// tokenizer reads the kanji alone (痛 → つう) and the result becomes 「つーーい」.
    func stretchedOkurigana(_ text: String) -> String {
        guard text.contains("ー") else {
            return text
        }
        let pattern = try? NSRegularExpression(pattern: "([\\p{Han}]+)ー([\\p{Hiragana}]+)")
        guard let regex = pattern else {
            return text
        }
        let ns = text as NSString
        var result = text
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).reversed() {
            let kanji = ns.substring(with: match.range(at: 1))
            let okurigana = ns.substring(with: match.range(at: 2))
            let reading = convertToHiragana(kanji + okurigana)
            var spelled = reading
            if reading.hasSuffix(okurigana) {
                spelled = String(reading.dropLast(okurigana.count)) + "ー" + okurigana
            }
            result = (result as NSString).replacingCharacters(in: match.range, with: spelled)
        }
        return result
    }

    /// Normalizes text to hiragana with punctuation (see `kanaOnly`).
    public func convertToHiragana(_ text: String) -> String {
        if text.isEmpty {
            return ""
        }

        var rawHira = ""
        for token in tokenize(text) {
            rawHira.append(token.reading)
        }

        return Self.tidyKana(kanaOnly(rawHira))
    }

    /// Converts katakana to hiragana.
    public func normalizeKana(_ text: String) -> String {
        if text.isEmpty {
            return ""
        }

        var normalized = ""
        for scalar in text.unicodeScalars {
            let val = scalar.value
            switch val {
            case 0x30A1...0x30F6:
                switch UnicodeScalar(val - 0x60) {
                case .some(let hScalar):
                    normalized.append(Character(hScalar))
                case .none:
                    normalized.append(Character(scalar))
                }
            case _ where Self.isProlongedMark(val):
                normalized.append("ー")
            default:
                normalized.append(Character(scalar))
            }
        }

        return normalized
    }
}
