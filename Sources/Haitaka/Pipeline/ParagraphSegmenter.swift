import Foundation

/// A paragraph of text with its start and end time.
public struct ParagraphSegment: Sendable, Equatable {
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let text: String

    public init(startTimeSeconds: Float, endTimeSeconds: Float, text: String) {
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.text = text
    }

    /// Formatted as "[00:01.20 -> 00:05.80]".
    public var timestampFormatted: String {
        ParagraphSegmenter.formatTimestampRange(start: startTimeSeconds, end: endTimeSeconds)
    }

    /// Timestamp range followed by the text, e.g. "[00:01.20 -> 00:05.80] 本日はお日柄もよく...".
    public var formattedLine: String {
        if text.isEmpty {
            return timestampFormatted
        }
        return "\(timestampFormatted) \(text)"
    }
}

/// Groups utterances into paragraphs, splitting at pauses of at least `pauseThresholdSeconds`,
/// or at 0.8 s or more after sentence-final punctuation.
public final class ParagraphSegmenter: @unchecked Sendable {
    public let pauseThresholdSeconds: Float

    private var currentStart: Float = 0.0
    private var currentEnd: Float = 0.0
    private var currentTextParts: [String] = []
    private var hasActiveParagraph: Bool = false

    public init(pauseThresholdSeconds: Float = 1.2) {
        self.pauseThresholdSeconds = pauseThresholdSeconds
    }

    /// Formats seconds as mm:ss.xx, or hh:mm:ss.xx from one hour.
    public static func formatTimestamp(seconds: Float) -> String {
        var sec = seconds
        if sec < 0.0 {
            sec = 0.0
        }
        let totalCentis = Int((sec * 100.0).rounded())
        let centis = totalCentis % 100
        let totalSecs = totalCentis / 100
        let s = totalSecs % 60
        let totalMins = totalSecs / 60
        let m = totalMins % 60
        let h = totalMins / 60

        if 0 < h {
            return String(format: "%02d:%02d:%02d.%02d", h, m, s, centis)
        }
        return String(format: "%02d:%02d.%02d", m, s, centis)
    }

    /// Formats a range as "[00:01.20 -> 00:05.80]".
    public static func formatTimestampRange(start: Float, end: Float) -> String {
        let sStr = formatTimestamp(seconds: start)
        let eStr = formatTimestamp(seconds: end)
        return "[\(sStr) -> \(eStr)]"
    }

    private static func isSentenceDelimiter(_ ch: Character) -> Bool {
        switch ch {
        case "。", "！", "？", ".", "!", "?":
            return true
        default:
            return false
        }
    }

    /// Adds an utterance. Returns the completed paragraph when this utterance starts a new one.
    public func appendSegment(text: String, startSeconds: Float, endSeconds: Float) -> ParagraphSegment? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return nil
        }

        if hasActiveParagraph {
            let pauseDuration = startSeconds - currentEnd

            var shouldSplit = false
            if pauseThresholdSeconds <= pauseDuration {
                shouldSplit = true
            } else {
                // Shorter pauses still split after sentence-final punctuation.
                if let lastPart = currentTextParts.last, let lastChar = lastPart.last {
                    if Self.isSentenceDelimiter(lastChar) && 0.8 <= pauseDuration {
                        shouldSplit = true
                    }
                }
            }

            if shouldSplit {
                let completed = buildCurrentParagraph()
                currentStart = startSeconds
                currentEnd = endSeconds
                currentTextParts = [trimmed]
                hasActiveParagraph = true
                return completed
            }

            if currentEnd < endSeconds {
                currentEnd = endSeconds
            }
            currentTextParts.append(trimmed)
        } else {
            currentStart = startSeconds
            currentEnd = endSeconds
            currentTextParts = [trimmed]
            hasActiveParagraph = true
        }

        return nil
    }

    private func buildCurrentParagraph() -> ParagraphSegment {
        let combinedText = currentTextParts.joined()
        return ParagraphSegment(
            startTimeSeconds: currentStart,
            endTimeSeconds: currentEnd,
            text: combinedText
        )
    }

    /// Returns the pending paragraph, if any, and resets.
    public func flush() -> ParagraphSegment? {
        if hasActiveParagraph != true || currentTextParts.isEmpty {
            return nil
        }
        let paragraph = buildCurrentParagraph()
        reset()
        return paragraph
    }

    public func reset() {
        currentStart = 0.0
        currentEnd = 0.0
        currentTextParts.removeAll(keepingCapacity: true)
        hasActiveParagraph = false
    }

    /// Segments a complete list of results into paragraphs.
    public static func segment(
        results: [TranscriptionResult],
        pauseThresholdSeconds: Float = 1.2
    ) -> [ParagraphSegment] {
        let segmenter = ParagraphSegmenter(pauseThresholdSeconds: pauseThresholdSeconds)
        var paragraphs: [ParagraphSegment] = []

        for res in results {
            if let completed = segmenter.appendSegment(
                text: res.text,
                startSeconds: res.startTimeSeconds,
                endSeconds: res.endTimeSeconds
            ) {
                paragraphs.append(completed)
            }
        }

        if let finalParagraph = segmenter.flush() {
            paragraphs.append(finalParagraph)
        }

        return paragraphs
    }
}

/// One recognized utterance with its time span.
public struct TranscriptionResult: Sendable, Equatable {
    public let text: String
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float

    public init(text: String, startTimeSeconds: Float, endTimeSeconds: Float) {
        self.text = text
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
    }
}
