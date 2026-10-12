import XCTest
@testable import Haitaka

final class ParagraphSegmenterTests: XCTestCase {

    func testTimestampFormatting() {
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 1.20), "00:01.20")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 5.80), "00:05.80")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 65.45), "01:05.45")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 3661.05), "01:01:01.05")

        let rangeStr = ParagraphSegmenter.formatTimestampRange(start: 1.20, end: 5.80)
        XCTAssertEqual(rangeStr, "[00:01.20 -> 00:05.80]")

        // Negative values clamp to zero.
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: -2.0), "00:00.00")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 0.0), "00:00.00")
    }

    func testPauseBasedSegmentation() {
        let segmenter = ParagraphSegmenter(pauseThresholdSeconds: 1.2)

        // [0.0 -> 2.0] opens the first paragraph; nothing is completed yet.
        let p1 = segmenter.appendSegment(text: "本日は晴天です", startSeconds: 0.0, endSeconds: 2.0)
        XCTAssertNil(p1)

        // [2.5 -> 4.0]: a 0.5 s pause is below 1.2 s, so it joins the same paragraph.
        let p2 = segmenter.appendSegment(text: "気温も過ごしやすいです", startSeconds: 2.5, endSeconds: 4.0)
        XCTAssertNil(p2)

        // [6.0 -> 8.5]: a 2.0 s pause exceeds 1.2 s and completes the previous paragraph.
        let completed = segmenter.appendSegment(text: "次の話題に移ります", startSeconds: 6.0, endSeconds: 8.5)
        XCTAssertNotNil(completed)

        if let comp = completed {
            XCTAssertEqual(comp.startTimeSeconds, 0.0)
            XCTAssertEqual(comp.endTimeSeconds, 4.0)
            XCTAssertEqual(comp.text, "本日は晴天です気温も過ごしやすいです")
            XCTAssertEqual(comp.timestampFormatted, "[00:00.00 -> 00:04.00]")
            XCTAssertEqual(comp.formattedLine, "[00:00.00 -> 00:04.00] 本日は晴天です気温も過ごしやすいです")
        }

        let finalP = segmenter.flush()
        XCTAssertNotNil(finalP)
        if let fp = finalP {
            XCTAssertEqual(fp.startTimeSeconds, 6.0)
            XCTAssertEqual(fp.endTimeSeconds, 8.5)
            XCTAssertEqual(fp.text, "次の話題に移ります")
            XCTAssertEqual(fp.timestampFormatted, "[00:06.00 -> 00:08.50]")
        }
    }

    func testSentenceDelimiterSegmentation() {
        let segmenter = ParagraphSegmenter(pauseThresholdSeconds: 1.5)

        let p1 = segmenter.appendSegment(text: "よろしくお願いします。", startSeconds: 1.0, endSeconds: 3.0)
        XCTAssertNil(p1)

        // A 0.9 s pause after 。 is at least 0.8 s, so it splits below the 1.5 s threshold.
        let completed = segmenter.appendSegment(text: "それでは始めます。", startSeconds: 3.9, endSeconds: 5.5)
        XCTAssertNotNil(completed)

        if let comp = completed {
            XCTAssertEqual(comp.text, "よろしくお願いします。")
            XCTAssertEqual(comp.timestampFormatted, "[00:01.00 -> 00:03.00]")
        }
    }

    func testBatchSegmentation() {
        let r1 = TranscriptionResult(
            text: "こんにちは。",
            startTimeSeconds: 1.20,
            endTimeSeconds: 3.50
        )
        let r2 = TranscriptionResult(
            text: "haitakaの紹介です。",
            startTimeSeconds: 3.80,
            endTimeSeconds: 5.80
        )
        let r3 = TranscriptionResult(
            text: "第2段落の内容です。",
            startTimeSeconds: 8.00,
            endTimeSeconds: 10.50
        )

        let paragraphs = ParagraphSegmenter.segment(results: [r1, r2, r3], pauseThresholdSeconds: 1.2)
        XCTAssertEqual(paragraphs.count, 2)

        XCTAssertEqual(paragraphs[0].timestampFormatted, "[00:01.20 -> 00:05.80]")
        XCTAssertEqual(paragraphs[0].text, "こんにちは。haitakaの紹介です。")
        XCTAssertEqual(paragraphs[0].formattedLine, "[00:01.20 -> 00:05.80] こんにちは。haitakaの紹介です。")

        XCTAssertEqual(paragraphs[1].timestampFormatted, "[00:08.00 -> 00:10.50]")
        XCTAssertEqual(paragraphs[1].text, "第2段落の内容です。")
    }

    func testEmptyAndReset() {
        let segmenter = ParagraphSegmenter()

        XCTAssertNil(segmenter.appendSegment(text: "", startSeconds: 0.0, endSeconds: 1.0))
        XCTAssertNil(segmenter.appendSegment(text: "   ", startSeconds: 0.0, endSeconds: 1.0))
        XCTAssertNil(segmenter.flush())

        let p1 = segmenter.appendSegment(text: "テスト", startSeconds: 0.0, endSeconds: 1.0)
        XCTAssertNil(p1)
        segmenter.reset()
        XCTAssertNil(segmenter.flush())
    }
}
