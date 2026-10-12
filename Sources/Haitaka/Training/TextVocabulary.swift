import Foundation

/// Character vocabulary (kanji, kana, symbols) with reserved special tokens pad, unk, sos, eos.
public struct TextVocabulary: Sendable {
    public static let padId = 0
    public static let unkId = 1
    public static let sosId = 2
    public static let eosId = 3

    public let size: Int
    private let tokenList: [Character]
    private let charToIdTable: [Character: Int]

    /// Builds the vocabulary from every character in the corpus.
    public init(corpus: [String] = []) {
        var set: Set<Character> = []
        var cIdx = 0
        while cIdx < corpus.count {
            let line = corpus[cIdx]
            for c in line {
                set.insert(c)
            }
            cIdx += 1
        }

        // Placeholders for pad(0), unk(1), sos(2), eos(3).
        let specialChars: [Character] = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        
        let sortedChars = Array(set).sorted { (a, b) -> Bool in
            a < b
        }

        var list = specialChars
        var sIdx = 0
        while sIdx < sortedChars.count {
            let c = sortedChars[sIdx]
            if specialChars.contains(c) != true {
                list.append(c)
            }
            sIdx += 1
        }

        self.size = list.count
        self.tokenList = list

        var table: [Character: Int] = [:]
        var i = 0
        while i < list.count {
            table[list[i]] = i
            i += 1
        }
        self.charToIdTable = table
    }

    public init(characters: [Character]) {
        let specialChars: [Character] = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        var list = specialChars
        var sIdx = 0
        while sIdx < characters.count {
            let c = characters[sIdx]
            if specialChars.contains(c) != true {
                list.append(c)
            }
            sIdx += 1
        }
        self.size = list.count
        self.tokenList = list

        var table: [Character: Int] = [:]
        var i = 0
        while i < list.count {
            table[list[i]] = i
            i += 1
        }
        self.charToIdTable = table
    }

    /// All characters except the special tokens. Stored with the weights; `init(serializedCharacters:)`
    /// restores the same ID assignment.
    public var serializedCharacters: String {
        let specialChars: Set<Character> = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        var result = ""
        var i = 0
        while i < tokenList.count {
            let c = tokenList[i]
            if specialChars.contains(c) != true {
                result.append(c)
            }
            i += 1
        }
        return result
    }

    public init(serializedCharacters: String) {
        self.init(characters: Array(serializedCharacters))
    }

    public func id(for char: Character) -> Int {
        switch charToIdTable[char] {
        case .some(let val):
            return val
        case .none:
            return Self.unkId
        }
    }

    public func char(for id: Int) -> Character {
        switch true {
        case id < 0:
            return "?"
        case tokenList.count <= id:
            return "?"
        case id == Self.padId:
            return " "
        case id == Self.unkId:
            return "?"
        case id == Self.sosId:
            return "^"
        case id == Self.eosId:
            return "$"
        default:
            return tokenList[id]
        }
    }

    public func textToIds(_ text: String) -> [Int] {
        var ids = [Int](repeating: 0, count: text.count)
        var i = 0
        for c in text {
            ids[i] = id(for: c)
            i += 1
        }
        return ids
    }

    public func idsToText(_ ids: [Int]) -> String {
        var result = ""
        var i = 0
        while i < ids.count {
            let idVal = ids[i]
            switch idVal {
            case Self.padId, Self.sosId, Self.eosId:
                break
            case Self.unkId:
                result.append("?")
            default:
                if 0 <= idVal && idVal < tokenList.count {
                    result.append(tokenList[idVal])
                }
            }
            i += 1
        }
        return result
    }
}
