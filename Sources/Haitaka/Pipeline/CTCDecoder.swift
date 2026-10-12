import Foundation

/// Trie that interns prefixes as integer node IDs, so lookup and extension are O(1)
/// instead of hashing and copying a whole `[Int]` prefix.
struct CTCPrefixTrie {
    var parents: [Int32] = [-1]
    var tokens: [Int32] = [-1]
    var depths: [Int32] = [0]
    /// LM score of the whole prefix (weight · LM log-prob + length bonus per character), kept separate from the acoustic probability.
    var scores: [Float] = [0.0]
    /// (parent node, token) -> child node
    var children: [Int64: Int32] = [:]

    /// Returns the node for `node` + `token`, creating it if needed. When `isNew` is true the caller must add the score.
    mutating func child(of node: Int32, token: Int) -> (node: Int32, isNew: Bool) {
        let key = (Int64(node) << 32) | Int64(token)
        switch children[key] {
        case .some(let existing):
            return (existing, false)
        case .none:
            let newNode = Int32(parents.count)
            parents.append(node)
            tokens.append(Int32(token))
            depths.append(depths[Int(node)] + 1)
            scores.append(scores[Int(node)])
            children[key] = newNode
            return (newNode, true)
        }
    }

    /// Last `count` tokens of the node's prefix, oldest first.
    func suffix(of node: Int32, count: Int) -> [Int] {
        var result: [Int] = []
        var cursor = node
        while 0 < cursor && result.count < count {
            result.append(Int(tokens[Int(cursor)]))
            cursor = parents[Int(cursor)]
        }
        result.reverse()
        return result
    }

    func sequence(of node: Int32) -> [Int] {
        var result = [Int](repeating: 0, count: Int(depths[Int(node)]))
        var cursor = node
        var pos = result.count - 1
        while 0 <= pos {
            result[pos] = Int(tokens[Int(cursor)])
            cursor = parents[Int(cursor)]
            pos -= 1
        }
        return result
    }

    mutating func removeAll() {
        parents = [-1]
        tokens = [-1]
        depths = [0]
        scores = [0.0]
        children.removeAll(keepingCapacity: true)
    }
}

/// Beam hypothesis identified by its trie node.
struct CTCNodeHypothesis {
    var node: Int32
    var pBlank: Float
    var pNonBlank: Float
    var totalProb: Float {
        logAdd(pBlank, pNonBlank)
    }
}

/// CTC prefix beam search that advances one frame at a time. Partial results are available from `best` at
/// any point; `reset()` starts a new utterance. With a kana LM, each appended character is scored by
/// shallow fusion: acoustic + `KanaNgramLM.fusionWeight` · log P(char) + `KanaNgramLM.fusionLengthBonus`.
public final class CTCDecoder: @unchecked Sendable {
    /// Non-blank tokens expanded per frame, taken from the most probable.
    public static let tokenPruneCount = 8

    public let vocabulary: TextVocabulary
    public let beamWidth: Int
    public let languageModel: KanaNgramLM?
    private let languageWeight: Float
    private let lengthBonus: Float

    /// Number of frames pushed since the last reset.
    public private(set) var frameCount = 0

    private var trie = CTCPrefixTrie()
    private var beams: [CTCNodeHypothesis]
    private var nextIndex: [Int32: Int] = [:]
    private var nextList: [CTCNodeHypothesis] = []
    private var candidateTokens: [Int] = []
    private var candidateScores: [Float] = []

    public init(vocabulary: TextVocabulary, beamWidth: Int = 16, languageModel: KanaNgramLM? = nil) {
        self.vocabulary = vocabulary
        self.beamWidth = beamWidth
        self.languageModel = languageModel
        var weight: Float = 0.0
        var bonus: Float = 0.0
        if languageModel != nil {
            weight = KanaNgramLM.fusionWeight
            bonus = KanaNgramLM.fusionLengthBonus
        }
        self.languageWeight = weight
        self.lengthBonus = bonus
        self.beams = [CTCNodeHypothesis(node: 0, pBlank: 0.0, pNonBlank: -.infinity)]
    }

    /// Clears the state. Call at utterance boundaries.
    public func reset() {
        trie.removeAll()
        beams = [CTCNodeHypothesis(node: 0, pBlank: 0.0, pNonBlank: -.infinity)]
        nextIndex.removeAll(keepingCapacity: true)
        nextList.removeAll(keepingCapacity: true)
        frameCount = 0
    }

    /// Advances the beam by one frame of log-probabilities.
    public func push(frame logProbs: [Float]) {
        let vCount = logProbs.count
        if vCount <= 1 {
            return
        }
        if candidateTokens.count < vCount {
            candidateTokens = [Int](repeating: 0, count: vCount)
            candidateScores = [Float](repeating: -.infinity, count: vCount)
        }

        nextIndex.removeAll(keepingCapacity: true)
        nextList.removeAll(keepingCapacity: true)

        // Expanding the full vocabulary costs frames × beam × V node operations; keep only the top tokenPruneCount tokens.
        var candidateCount = 0
        var c = 1
        while c < vCount {
            let p = logProbs[c]
            if candidateCount < Self.tokenPruneCount {
                var insertAt = candidateCount
                while 0 < insertAt && candidateScores[insertAt - 1] < p {
                    candidateScores[insertAt] = candidateScores[insertAt - 1]
                    candidateTokens[insertAt] = candidateTokens[insertAt - 1]
                    insertAt -= 1
                }
                candidateScores[insertAt] = p
                candidateTokens[insertAt] = c
                candidateCount += 1
            } else {
                if candidateScores[candidateCount - 1] < p {
                    var insertAt = candidateCount - 1
                    while 0 < insertAt && candidateScores[insertAt - 1] < p {
                        candidateScores[insertAt] = candidateScores[insertAt - 1]
                        candidateTokens[insertAt] = candidateTokens[insertAt - 1]
                        insertAt -= 1
                    }
                    candidateScores[insertAt] = p
                    candidateTokens[insertAt] = c
                }
            }
            c += 1
        }

        let pBlankCurr = logProbs[TextVocabulary.padId]
        var bi = 0
        while bi < beams.count {
            let hyp = beams[bi]

            // Blank: prefix unchanged.
            let sameSlot = slot(for: hyp.node)
            nextList[sameSlot].pBlank = logAdd(
                nextList[sameSlot].pBlank, hyp.totalProb + pBlankCurr)

            // Non-blank: expand pruned candidates only.
            let endToken = Int(trie.tokens[Int(hyp.node)])
            var ci = 0
            while ci < candidateCount {
                let token = candidateTokens[ci]
                let pChar = logProbs[token]
                let (childNode, isNew) = trie.child(of: hyp.node, token: token)
                if isNew {
                    trie.scores[Int(childNode)] += languageScore(parent: hyp.node, token: token)
                }
                if token == endToken {
                    // Repeated token: from pBlank it appends a new character.
                    let newSlot = slot(for: childNode)
                    nextList[newSlot].pNonBlank = logAdd(
                        nextList[newSlot].pNonBlank, hyp.pBlank + pChar)

                    // From pNonBlank it continues the same character (prefix unchanged).
                    let contSlot = slot(for: hyp.node)
                    nextList[contSlot].pNonBlank = logAdd(
                        nextList[contSlot].pNonBlank, hyp.pNonBlank + pChar)
                } else {
                    let newSlot = slot(for: childNode)
                    nextList[newSlot].pNonBlank = logAdd(
                        nextList[newSlot].pNonBlank, hyp.totalProb + pChar)
                }
                ci += 1
            }
            bi += 1
        }

        // Keep the top beamWidth by acoustic + LM score. Ties break by node ID for deterministic output.
        nextList.sort { a, b in
            let sa = a.totalProb + trie.scores[Int(a.node)]
            let sb = b.totalProb + trie.scores[Int(b.node)]
            if sa != sb {
                return sb < sa
            }
            return a.node < b.node
        }

        beams.removeAll(keepingCapacity: true)
        var k = 0
        let limit = min(beamWidth, nextList.count)
        while k < limit {
            beams.append(nextList[k])
            k += 1
        }
        frameCount += 1
    }

    public func push(frames logProbsSeq: [[Float]]) {
        var t = 0
        while t < logProbsSeq.count {
            push(frame: logProbsSeq[t])
            t += 1
        }
    }

    /// Decodes a whole utterance from scratch.
    public func decode(logProbs: [[Float]]) -> (text: String, tokens: [Int], score: Float) {
        reset()
        push(frames: logProbs)
        return best
    }

    /// Current best hypothesis. `score` is the acoustic log-probability without the LM score.
    public var best: (text: String, tokens: [Int], score: Float) {
        var bestNode: Int32 = -1
        var bestTotal: Float = -.infinity
        var bestScore: Float = -.infinity

        var i = 0
        while i < beams.count {
            let hyp = beams[i]
            let total = hyp.totalProb
            let score = total + trie.scores[Int(hyp.node)]
            if bestScore < score {
                bestScore = score
                bestTotal = total
                bestNode = hyp.node
            }
            i += 1
        }

        if bestNode < 0 {
            return ("", [], -.infinity)
        }
        let tokens = trie.sequence(of: bestNode)
        return (vocabulary.idsToText(tokens), tokens, bestTotal)
    }

    /// LM score increment for appending `token` to `parent`.
    private func languageScore(parent: Int32, token: Int) -> Float {
        var score = lengthBonus
        if let lm = languageModel {
            score += languageWeight * lm.logProb(token: token, history: trie.suffix(of: parent, count: lm.order - 1))
        }
        return score
    }

    /// Index of the node's hypothesis in nextList, creating it with -inf probabilities if absent.
    private func slot(for node: Int32) -> Int {
        switch nextIndex[node] {
        case .some(let idx):
            return idx
        case .none:
            let idx = nextList.count
            nextList.append(CTCNodeHypothesis(node: node, pBlank: -.infinity, pNonBlank: -.infinity))
            nextIndex[node] = idx
            return idx
        }
    }
}

/// log(exp(a) + exp(b))
@inline(__always)
func logAdd(_ a: Float, _ b: Float) -> Float {
    if a == -.infinity {
        return b
    }
    if b == -.infinity {
        return a
    }
    if a < b {
        return b + log(1.0 + exp(a - b))
    }
    return a + log(1.0 + exp(b - a))
}
