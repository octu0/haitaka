import Foundation

public enum FillerFilterMode: Sendable, Equatable {
    case remove
    case mark         // wrap fillers in parentheses, e.g. "(えー)"
    case disabled
}

/// Removes or marks filler words (えー, あのー, ...) in recognized text.
public struct FillerWordFilter: Sendable {
    public let mode: FillerFilterMode

    public init(mode: FillerFilterMode = .remove) {
        self.mode = mode
    }

    /// Maps wave dashes and similar lengthening marks to "ー".
    private static func normalizeProlongedMarks(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x301C, // WAVE DASH (〜)
                 0xFF5E, // FULLWIDTH TILDE (～)
                 0x3030, // WAVY DASH (〰)
                 0x223C, // TILDE OPERATOR (∼)
                 0xFF70: // HALFWIDTH KATAKANA-HIRAGANA PROLONGED SOUND MARK (ｰ)
                result.append("ー")
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    /// Punctuation, whitespace, symbols and brackets.
    private static func isDelimiter(_ ch: Character) -> Bool {
        switch ch {
        case "、", "。", "，", "．", " ", "　", "\t", "\n", "\r",
             "！", "？", "!", "?", "…", "・",
             "「", "」", "『", "』", "（", "）", "(", ")", "【", "】", "\"", "'":
            return true
        default:
            return false
        }
    }

    /// CJK Unified Ideographs and Extension A.
    private static func isKanji(_ ch: Character) -> Bool {
        for scalar in ch.unicodeScalars {
            if 0x4E00 <= scalar.value && scalar.value <= 0x9FFF {
                return true
            }
            if 0x3400 <= scalar.value && scalar.value <= 0x4DBF {
                return true
            }
        }
        return false
    }

    /// Whether the whole string is a filler word.
    public func isFiller(_ text: String) -> Bool {
        let normalized = Self.normalizeProlongedMarks(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if normalized.isEmpty {
            return false
        }

        switch normalized {
        case "えー", "えーー", "えーーー", "ええと", "えっと", "えっとー", "えーと", "えーっと", "ええ",
             "あー", "あーー", "あーーー", "あのー", "あのーー", "あのね", "あのーっ", "あーっと", "あっ",
             "そのー", "そのーー", "そのーっ",
             "うーん", "うーーん", "うーんと", "うむ", "んー", "んーと",
             "まあ", "まー", "まーっ",
             "なんか", "なんかー", "なんというか",
             "あの", "その",
             "ほら", "ほらー":
            return true
        default:
            return false
        }
    }

    /// Removes or marks fillers in `text` according to `mode`.
    public func filter(_ text: String) -> String {
        switch mode {
        case .disabled:
            return text
        case .remove, .mark:
            break
        }

        if text.isEmpty {
            return ""
        }

        let normalized = Self.normalizeProlongedMarks(text)
        let chars = Array(normalized)
        let n = chars.count

        var output = ""
        output.reserveCapacity(n)

        var i = 0
        while i < n {
            var matchedLength = 0
            var matchedText = ""

            // Longest match first; the longest filler is 6 characters (なんというか).
            let maxCheckLen = min(6, n - i)
            var l = maxCheckLen
            while 1 <= l {
                let subStr = String(chars[i..<(i + l)])
                var isCandidate = false
                var requireDelimiterOrEnd = false

                switch subStr {
                case "なんというか", "えーっと", "あーっと", "うーんと", "えっとー", "あのねー",
                     "えーと", "ええと", "えっと", "あのー", "そのー", "うーん", "なんかー", "ほらー",
                     "えー", "あー", "まーっ", "あのーっ", "そのーっ",
                     "うむ", "んー", "まあ", "まー", "なんか":
                    isCandidate = true
                case "あの", "その":
                    // Demonstratives: fillers only before a delimiter or the end, so あの人 is kept.
                    isCandidate = true
                    requireDelimiterOrEnd = true
                default:
                    isCandidate = false
                }

                if isCandidate {
                    var isValid = true
                    let nextIdx = i + l

                    if requireDelimiterOrEnd {
                        if nextIdx < n {
                            let nextCh = chars[nextIdx]
                            if Self.isDelimiter(nextCh) != true {
                                // Also accept when another filler follows (あのえー, そのうーん).
                                let nextRemain = n - nextIdx
                                var followedByFiller = false
                                if 2 <= nextRemain {
                                    let nextSub2 = String(chars[nextIdx..<(nextIdx + 2)])
                                    switch nextSub2 {
                                    case "えっ", "えー", "うー", "あー", "その", "あの", "まー":
                                        followedByFiller = true
                                    default:
                                        break
                                    }
                                }
                                if followedByFiller != true {
                                    isValid = false
                                }
                            }
                        }
                    }

                    // Make sure あー/えー/まー are not part of a word (あーと, まーけっと, えーす).
                    if isValid {
                        switch subStr {
                        case "あー", "えー", "まー":
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                // Followed by kana other than "ー": part of a word unless another filler starts there.
                                if Self.isDelimiter(nextCh) != true && Self.isKanji(nextCh) != true && nextCh != "ー" {
                                    let nextRemain = n - nextIdx
                                    var followedByFiller = false
                                    if 2 <= nextRemain {
                                        let nextSub2 = String(chars[nextIdx..<(nextIdx + 2)])
                                        switch nextSub2 {
                                        case "えっ", "あの", "その", "えー", "うー":
                                            followedByFiller = true
                                        default:
                                            break
                                        }
                                    }
                                    if followedByFiller != true {
                                        isValid = false
                                    }
                                }
                            }
                        case "まあ":
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                // Filler only before a delimiter, kanji or the end, so まあまあ is kept.
                                if Self.isDelimiter(nextCh) != true && Self.isKanji(nextCh) != true {
                                    isValid = false
                                }
                            }
                        case "なんか":
                            if nextIdx < n {
                                let nextCh = chars[nextIdx]
                                // なんか + case particle (なんかが, なんかを) is the pronoun "something", not a filler.
                                switch nextCh {
                                case "が", "を", "に", "で", "と", "の":
                                    isValid = false
                                default:
                                    if Self.isDelimiter(nextCh) != true && Self.isKanji(nextCh) != true {
                                        isValid = false
                                    }
                                }
                            }
                        default:
                            break
                        }
                    }

                    if isValid {
                        // Absorb trailing "ー" (えーーー).
                        var ext = nextIdx
                        while ext < n {
                            if chars[ext] == "ー" {
                                ext += 1
                            } else {
                                break
                            }
                        }
                        matchedLength = ext - i
                        matchedText = String(chars[i..<ext])
                        break
                    }
                }

                l -= 1
            }

            if 0 < matchedLength {
                switch mode {
                case .remove:
                    // Also drop "、" or spaces before the filler.
                    while let last = output.last, last == "、" || last == " " || last == "　" {
                        output.removeLast()
                    }
                case .mark:
                    output.append("(\(matchedText))")
                case .disabled:
                    ()
                }

                i += matchedLength

                // Skip "、" or spaces after the filler.
                skipDelimiters: while i < n {
                    switch chars[i] {
                    case "、", " ", "　":
                        i += 1
                    default:
                        break skipDelimiters
                    }
                }
            } else {
                output.append(chars[i])
                i += 1
            }
        }

        // Strip leading punctuation and whitespace left behind.
        var slice = output[...]
        trimLeading: while let first = slice.first {
            switch first {
            case "、", "。", " ", "　", "\t":
                slice = slice.dropFirst()
            default:
                break trimLeading
            }
        }

        return String(slice)
    }
}
