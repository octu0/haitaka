import Foundation

/// Builds katakana-English kana readings for English words from a CMU-format pronouncing dictionary (`word PH PH ...`, ARPAbet phones).
///
/// Transcripts keep English as written (song lyrics, "YouTube"). The acoustic model is trained on kana,
/// so English phones are mapped onto Japanese sounds. The written form stays English in the stage-2 dictionary.
public final class EnglishPronunciations: @unchecked Sendable {
    private let phones: [String: [String]]

    /// Loads a CMU-format dictionary file.
    public convenience init(contentsOfFile path: String) throws {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        self.init(lines: content.split(separator: "\n").map(String.init))
    }

    public var count: Int {
        return phones.count
    }

    /// Parses `word PH PH ...` lines. Alternate pronunciations (`word(2)`) are skipped; text after `#` is ignored. Stress digits are dropped.
    public init(lines: [String]) {
        var table: [String: [String]] = [:]
        table.reserveCapacity(lines.count)
        for rawLine in lines {
            var line = rawLine
            if let hash = line.firstIndex(of: "#") {
                line = String(line[..<hash])
            }
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            if parts.count < 2 {
                continue
            }
            let word = String(parts[0]).lowercased()
            if word.hasSuffix(")") {
                continue
            }
            var stripped: [String] = []
            stripped.reserveCapacity(parts.count - 1)
            var i = 1
            while i < parts.count {
                stripped.append(String(parts[i].filter { $0.isNumber != true }))
                i += 1
            }
            if table[word] == nil {
                table[word] = stripped
            }
        }
        self.phones = table
    }

    /// Kana reading of an English word, or nil if it is not in the dictionary.
    public func reading(of word: String) -> String? {
        guard let ph = phones[word.lowercased()] else {
            return nil
        }
        return Self.kana(fromPhones: ph)
    }

    // MARK: - ARPAbet to kana

    /// Vowel phone -> (nucleus a/i/u/e/o, tail: long mark or second element of a diphthong).
    static let vowels: [String: (Character, String)] = [
        "AA": ("a", ""), "AE": ("a", ""), "AH": ("a", ""), "AO": ("o", ""), "AW": ("a", "う"),
        "AY": ("a", "い"), "EH": ("e", ""), "ER": ("a", "ー"), "EY": ("e", "ー"), "IH": ("i", ""),
        "IY": ("i", "ー"), "OW": ("o", "ー"), "OY": ("o", "い"), "UH": ("u", ""), "UW": ("u", "ー"),
    ]

    /// Short vowels; a following stop becomes a geminate っ.
    static let shortVowels: Set<String> = ["AA", "AE", "AH", "AO", "EH", "IH", "UH"]

    /// Consonant rows in a, i, u, e, o order.
    static let rows: [String: [String]] = [
        "B": ["ば", "び", "ぶ", "べ", "ぼ"], "CH": ["ちゃ", "ち", "ちゅ", "ちぇ", "ちょ"],
        "D": ["だ", "でぃ", "どぅ", "で", "ど"], "DH": ["ざ", "じ", "ず", "ぜ", "ぞ"],
        "F": ["ふぁ", "ふぃ", "ふ", "ふぇ", "ふぉ"], "G": ["が", "ぎ", "ぐ", "げ", "ご"],
        "HH": ["は", "ひ", "ふ", "へ", "ほ"], "JH": ["じゃ", "じ", "じゅ", "じぇ", "じょ"],
        "K": ["か", "き", "く", "け", "こ"], "L": ["ら", "り", "る", "れ", "ろ"],
        "M": ["ま", "み", "む", "め", "も"], "N": ["な", "に", "ぬ", "ね", "の"],
        "NG": ["んが", "んぎ", "んぐ", "んげ", "んご"], "P": ["ぱ", "ぴ", "ぷ", "ぺ", "ぽ"],
        "R": ["ら", "り", "る", "れ", "ろ"], "S": ["さ", "し", "す", "せ", "そ"],
        "SH": ["しゃ", "し", "しゅ", "しぇ", "しょ"], "T": ["た", "てぃ", "とぅ", "て", "と"],
        "TH": ["さ", "し", "す", "せ", "そ"], "V": ["ゔぁ", "ゔぃ", "ゔ", "ゔぇ", "ゔぉ"],
        "W": ["わ", "うぃ", "う", "うぇ", "うぉ"], "Y": ["や", "い", "ゆ", "いぇ", "よ"],
        "Z": ["ざ", "じ", "ず", "ぜ", "ぞ"], "ZH": ["じゃ", "じ", "じゅ", "じぇ", "じょ"],
    ]

    /// Small-kana suffix for consonant + Y + vowel (tube = T Y UW), appended to the i-row kana.
    static let palatalTails: [Character: String] = ["a": "ゃ", "i": "", "u": "ゅ", "e": "ぇ", "o": "ょ"]

    /// Vowel inserted after a consonant with no following vowel (word-final or in a cluster).
    static func epenthetic(_ consonant: String) -> String {
        switch consonant {
        case "T":
            return "と"
        case "D":
            return "ど"
        case "N":
            return "ん"
        case "M":
            return "む"
        case "NG":
            return "んぐ"
        case "L":
            return "る"
        case "R":
            return "ー"
        case "CH":
            return "ち"
        case "JH":
            return "じ"
        case "SH":
            return "しゅ"
        case "ZH":
            return "じゅ"
        case "W", "Y":
            return ""
        default:
            if let row = rows[consonant] {
                return row[2]
            }
            return ""
        }
    }

    static func vowelIndex(_ v: Character) -> Int {
        switch v {
        case "a":
            return 0
        case "i":
            return 1
        case "u":
            return 2
        case "e":
            return 3
        default:
            return 4
        }
    }

    /// AA (pot, star) maps to お (pop → ぽっぷ, hot → ほっと),
    /// but to あ before R (star → すたー) or after W (wanna → わな).
    static func vowelNucleus(_ vowel: String, next: String?, consonant: String?, fallback: Character) -> Character {
        if vowel != "AA" {
            return fallback
        }
        if next == "R" || consonant == "W" {
            return "a"
        }
        return "o"
    }

    static func plainVowel(_ v: Character) -> String {
        return ["あ", "い", "う", "え", "お"][vowelIndex(v)]
    }

    /// Maps an ARPAbet phone sequence to kana.
    /// A consonant joins the following vowel into one kana; a consonant with no vowel gets an epenthetic vowel.
    /// A stop after a short vowel, at word end or before a consonant, is geminated (pop → ぽっぷ).
    /// R after a vowel becomes a long mark (star → すたー).
    public static func kana(fromPhones phones: [String]) -> String {
        var out = ""
        var i = 0
        var prevVowel: String? = nil
        while i < phones.count {
            let p = phones[i]
            if let (nucleus, tail) = vowels[p] {
                out += plainVowel(vowelNucleus(p, next: i + 1 < phones.count ? phones[i + 1] : nil, consonant: nil, fallback: nucleus))
                appendTail(&out, tail)
                prevVowel = p
                i += 1
                continue
            }
            // K/G after NG is absorbed into ん (thank → さんく).
            if p == "NG" && i + 1 < phones.count && (phones[i + 1] == "K" || phones[i + 1] == "G") {
                out += "ん"
                prevVowel = nil
                i += 1
                continue
            }
            var j = i + 1
            var palatal = false
            if j < phones.count && phones[j] == "Y" && j + 1 < phones.count && vowels[phones[j + 1]] != nil && p != "Y" {
                palatal = true
                j += 1
            }
            if j < phones.count, let (rawNucleus, tail) = vowels[phones[j]] {
                let nucleus = vowelNucleus(phones[j], next: j + 1 < phones.count ? phones[j + 1] : nil, consonant: p, fallback: rawNucleus)
                if palatal {
                    // Palatal onset uses the i-row kana; T/D become ち/じ rather than てぃ/でぃ (tube → ちゅー).
                    let base: String
                    switch p {
                    case "T":
                        base = "ち"
                    case "D":
                        base = "じ"
                    default:
                        base = rows[p]?[1] ?? ""
                    }
                    out += base + (palatalTails[nucleus] ?? "")
                } else if (p == "K" || p == "G") && phones[j] == "AE" {
                    // K/G + AE is written as a palatal in katakana English (cat → きゃっと, gap → ぎゃっぷ).
                    out += (rows[p]?[1] ?? "") + "ゃ"
                } else if let row = rows[p] {
                    out += row[vowelIndex(nucleus)]
                } else {
                    out += plainVowel(nucleus)
                }
                appendTail(&out, tail)
                prevVowel = phones[j]
                i = j + 1
                continue
            }
            // Consonant with no following vowel.
            if p == "R" {
                if prevVowel != nil {
                    appendTail(&out, "ー")
                } else {
                    out += "る"
                }
            } else {
                let isStop = ["P", "T", "K", "B", "D", "G", "CH"].contains(p)
                if isStop, let pv = prevVowel, shortVowels.contains(pv) {
                    out += "っ"
                }
                out += epenthetic(p)
            }
            prevVowel = nil
            i += 1
        }
        return out
    }

    /// Appends a tail, avoiding a doubled long mark.
    static func appendTail(_ out: inout String, _ tail: String) {
        if tail == "ー" && out.hasSuffix("ー") {
            return
        }
        out += tail
    }
}
