import XCTest
@testable import Haitaka

final class FillerWordFilterTests: XCTestCase {

    func testStandaloneFillerRemoval() {
        let filter = FillerWordFilter(mode: .remove)

        XCTAssertEqual(filter.filter("えー"), "")
        XCTAssertEqual(filter.filter("あー"), "")
        XCTAssertEqual(filter.filter("えっと"), "")
        XCTAssertEqual(filter.filter("あのー"), "")
        XCTAssertEqual(filter.filter("そのー"), "")
        XCTAssertEqual(filter.filter("まあ"), "")
        XCTAssertEqual(filter.filter("なんか"), "")
        XCTAssertEqual(filter.filter("うーん"), "")
        XCTAssertEqual(filter.filter("ええと"), "")
        XCTAssertEqual(filter.filter("えーと"), "")
        XCTAssertEqual(filter.filter("えーっと"), "")
        XCTAssertEqual(filter.filter("なんというか"), "")
    }

    func testProlongedMarksVariations() {
        let filter = FillerWordFilter(mode: .remove)

        XCTAssertEqual(filter.filter("えーー"), "")
        XCTAssertEqual(filter.filter("えーーー"), "")
        XCTAssertEqual(filter.filter("あ〜"), "")
        XCTAssertEqual(filter.filter("あ〜〜"), "")
        XCTAssertEqual(filter.filter("う〜ん"), "")
        XCTAssertEqual(filter.filter("その〜"), "")
        XCTAssertEqual(filter.filter("あの〜"), "")
        XCTAssertEqual(filter.filter("まーっ"), "")
    }

    func testSententialFillerRemoval() {
        let filter = FillerWordFilter(mode: .remove)

        XCTAssertEqual(filter.filter("えー本日はお日柄もよく"), "本日はお日柄もよく")
        XCTAssertEqual(filter.filter("えー、本日はお日柄もよく"), "本日はお日柄もよく")
        XCTAssertEqual(filter.filter("あーえっと、明日は雨です"), "明日は雨です")
        XCTAssertEqual(filter.filter("そのー、15円です"), "15円です")

        XCTAssertEqual(filter.filter("明日の天気は、えー、晴れです"), "明日の天気は晴れです")
        XCTAssertEqual(filter.filter("これは、うーん、難しい問題です"), "これは難しい問題です")

        XCTAssertEqual(filter.filter("えー、あのー、そのー、会議を始めます"), "会議を始めます")

        XCTAssertEqual(filter.filter("えー、あのー、えっと"), "")
    }

    func testDemonstrativeProtection() {
        let filter = FillerWordFilter(mode: .remove)

        // あの and その before a noun are demonstratives, not fillers.
        XCTAssertEqual(filter.filter("あの人を見かけました"), "あの人を見かけました")
        XCTAssertEqual(filter.filter("その本を読みました"), "その本を読みました")
        XCTAssertEqual(filter.filter("あの店に行きましょう"), "あの店に行きましょう")

        // あの and その followed by a comma or pause are fillers.
        XCTAssertEqual(filter.filter("あの、ちょっといいですか"), "ちょっといいですか")
        XCTAssertEqual(filter.filter("その、実はですね"), "実はですね")
        XCTAssertEqual(filter.filter("あのー、あの人は誰ですか"), "あの人は誰ですか")
    }

    func testFilterModes() {
        let removeFilter = FillerWordFilter(mode: .remove)
        let markFilter = FillerWordFilter(mode: .mark)
        let disabledFilter = FillerWordFilter(mode: .disabled)

        let text = "えー、明日は晴れです"

        XCTAssertEqual(removeFilter.filter(text), "明日は晴れです")
        XCTAssertEqual(markFilter.filter(text), "(えー)明日は晴れです")
        XCTAssertEqual(disabledFilter.filter(text), "えー、明日は晴れです")
    }

    func testEdgeCases() {
        let filter = FillerWordFilter(mode: .remove)

        XCTAssertEqual(filter.filter(""), "")
        XCTAssertEqual(filter.filter("、"), "")
        XCTAssertEqual(filter.filter("   "), "")
        XCTAssertEqual(filter.filter("こんにちは、世界！"), "こんにちは、世界！")
        XCTAssertEqual(filter.filter("AIエンジンの高速化を達成しました"), "AIエンジンの高速化を達成しました")
    }

    func testLoanwordAndGrammarProtection() {
        let filter = FillerWordFilter(mode: .remove)

        // Hiragana loanwords keep their initial long vowel.
        XCTAssertEqual(filter.filter("あーとを鑑賞する"), "あーとを鑑賞する")
        XCTAssertEqual(filter.filter("まーけっとで買い物"), "まーけっとで買い物")
        XCTAssertEqual(filter.filter("えーすを狙う"), "えーすを狙う")

        // まあまあ and 何か are not fillers.
        XCTAssertEqual(filter.filter("まあまあです"), "まあまあです")
        XCTAssertEqual(filter.filter("なんかが起こった"), "なんかが起こった")
        XCTAssertEqual(filter.filter("何かありましたか"), "何かありましたか")

        // A filler directly before kanji is removed.
        XCTAssertEqual(filter.filter("あー本日は晴天です"), "本日は晴天です")
        XCTAssertEqual(filter.filter("えー田中です"), "田中です")
        XCTAssertEqual(filter.filter("まあ良いでしょう"), "良いでしょう")
    }

    func testBracketAndConsecutiveFillers() {
        let filter = FillerWordFilter(mode: .remove)

        // Fillers inside brackets.
        XCTAssertEqual(filter.filter("「あの、ちょっといいですか」"), "「ちょっといいですか」")
        XCTAssertEqual(filter.filter("「あの」"), "「」")

        // Consecutive fillers without a comma.
        XCTAssertEqual(filter.filter("あのえーっと、本日は晴天です"), "本日は晴天です")
        XCTAssertEqual(filter.filter("そのうーん、難しいです"), "難しいです")

        // A period right after a filler is removed with it.
        XCTAssertEqual(filter.filter("えー。"), "")
        XCTAssertEqual(filter.filter("えー。本日は晴天です"), "本日は晴天です")
    }
}
