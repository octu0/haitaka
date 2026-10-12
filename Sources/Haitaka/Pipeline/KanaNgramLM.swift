import Foundation

/// Kana character n-gram LM with interpolated Kneser-Ney smoothing, used for shallow fusion in CTC beam search.
/// Token IDs come from `TextVocabulary` and must be below 128 (7 bits each, packed into a UInt64 key).
/// Lines are left-padded with `TextVocabulary.sosId`, so sentence-initial characters have a context.
public final class KanaNgramLM: @unchecked Sendable {
    public let order: Int
    public let vocabularySize: Int
    /// Fusion score per appended character: fusionWeight · log P(char) + fusionLengthBonus.
    /// The length bonus offsets the LM's preference for short hypotheses, which would increase deletions.
    public static let defaultOrder = 5
    public static let fusionWeight: Float = 0.35
    public static let fusionLengthBonus: Float = 1.5
    /// Absolute discount.
    public static let discount: Float = 0.75
    private static let bitsPerToken = 7
    private static let tokenMask: UInt64 = 0x7F

    /// n-gram counts indexed by n - 1. Highest order holds raw counts; lower orders hold continuation counts (distinct left contexts).
    private var counts: [[UInt64: UInt32]]
    /// Per order: total count and number of distinct successors for each context of length n - 1.
    private var totals: [[UInt64: UInt32]]
    private var types: [[UInt64: UInt32]]

    public init(lines: [[Int]], order: Int, vocabularySize: Int) {
        precondition(1 <= order && order <= 8, "order must be 1...8")
        precondition(vocabularySize <= 128, "vocabulary must have at most 128 characters")
        self.order = order
        self.vocabularySize = vocabularySize
        counts = [[UInt64: UInt32]](repeating: [:], count: order)
        totals = [[UInt64: UInt32]](repeating: [:], count: order)
        types = [[UInt64: UInt32]](repeating: [:], count: order)

        let bos = TextVocabulary.sosId
        var top: [UInt64: UInt32] = [:]
        for line in lines {
            var key: UInt64 = 0
            var filled = 0
            var i = 0
            while i < order - 1 {
                key = push(key, bos, length: filled)
                filled += 1
                i += 1
            }
            for id in line {
                key = push(key, id, length: filled)
                filled = min(order, filled + 1)
                if filled == order {
                    top[key, default: 0] += 1
                }
            }
        }
        counts[order - 1] = top
        // Continuation counts: each distinct (n+1)-gram adds 1 to its suffix with the oldest character dropped.
        var n = order - 1
        while 1 <= n {
            var cont: [UInt64: UInt32] = [:]
            for key in counts[n].keys {
                cont[key >> UInt64(Self.bitsPerToken), default: 0] += 1
            }
            counts[n - 1] = cont
            n -= 1
        }
        n = 1
        while n <= order {
            let contextMask: UInt64 = n == 1 ? 0 : (UInt64(1) << UInt64(Self.bitsPerToken * (n - 1))) - 1
            for (key, c) in counts[n - 1] {
                let context = key & contextMask
                totals[n - 1][context, default: 0] += c
                types[n - 1][context, default: 0] += 1
            }
            n += 1
        }
    }

    /// Appends a token to a packed key (oldest token in the low bits). `length` is the current length; at full order the oldest token is dropped.
    @inline(__always)
    private func push(_ key: UInt64, _ id: Int, length: Int) -> UInt64 {
        let token = UInt64(id) & Self.tokenMask
        if length < order {
            return key | token << UInt64(Self.bitsPerToken * length)
        }
        return (key >> UInt64(Self.bitsPerToken)) | token << UInt64(Self.bitsPerToken * (order - 1))
    }

    /// Log-probability of `token` after `history` (oldest first; only the last order - 1 tokens are used, missing ones are treated as line start).
    public func logProb(token: Int, history: [Int]) -> Float {
        let bos = TextVocabulary.sosId
        var key: UInt64 = 0
        var len = 0
        let need = order - 1
        var i = history.count - need
        while i < history.count {
            var id = bos
            if 0 <= i {
                id = history[i]
            }
            key = key | (UInt64(id) & Self.tokenMask) << UInt64(Self.bitsPerToken * len)
            len += 1
            i += 1
        }
        return log(prob(token: token, contextKey: key, contextLength: len))
    }

    private func prob(token: Int, contextKey: UInt64, contextLength: Int) -> Float {
        let d = Self.discount
        let v = Float(vocabularySize)
        // Unigram from continuation counts, interpolated with uniform.
        var p: Float = 1.0 / v
        let unigramTotal = Float(totals[0][0] ?? 0)
        if 0 < unigramTotal {
            let c = Float(counts[0][UInt64(token) & Self.tokenMask] ?? 0)
            let t = Float(types[0][0] ?? 0)
            p = max(c - d, 0.0) / unigramTotal + d * t / unigramTotal * p
        }
        // Extend the context one token at a time; the newest m tokens are the high bits.
        var m = 1
        while m <= contextLength && m < order {
            let shift = UInt64(Self.bitsPerToken * (contextLength - m))
            let context = contextKey >> shift
            let total = Float(totals[m][context] ?? 0)
            if 0 < total {
                let ngram = context | (UInt64(token) & Self.tokenMask) << UInt64(Self.bitsPerToken * m)
                let c = Float(counts[m][ngram] ?? 0)
                let t = Float(types[m][context] ?? 0)
                p = max(c - d, 0.0) / total + d * t / total * p
            }
            m += 1
        }
        return p
    }

    /// Builds the LM from kana sentences, one per element.
    public convenience init(kanaLines: [String], vocabulary: TextVocabulary, order: Int = defaultOrder) {
        self.init(lines: kanaLines.map { vocabulary.textToIds($0) }, order: order, vocabularySize: vocabulary.size)
    }

    /// Builds the LM from a text file with one kana sentence per line (the output of `kanalm`).
    public convenience init?(contentsOfFile path: String, vocabulary: TextVocabulary) {
        guard let data = FileManager.default.contents(atPath: path), let whole = String(data: data, encoding: .utf8) else {
            return nil
        }
        self.init(kanaLines: whole.split(separator: "\n").map { String($0) }, vocabulary: vocabulary)
    }
}
