import XCTest
@testable import Haitaka

final class InverseTextNormalizerTests: XCTestCase {

    func testCurrencyNormalization() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("じゅうごえん"), "15円")
        XCTAssertEqual(itn.normalize("さんびゃくえん"), "300円")
        XCTAssertEqual(itn.normalize("せんごひゃくえん"), "1500円")
        XCTAssertEqual(itn.normalize("いちまんえん"), "10000円")
        XCTAssertEqual(itn.normalize("ごじゅうえん"), "50円")
        XCTAssertEqual(itn.normalize("いちえん"), "1円")

        XCTAssertEqual(itn.normalize("十五円"), "15円")
        XCTAssertEqual(itn.normalize("三百円"), "300円")
        XCTAssertEqual(itn.normalize("千五百円"), "1500円")
        XCTAssertEqual(itn.normalize("一万円"), "10000円")
        XCTAssertEqual(itn.normalize("五十円"), "50円")

        XCTAssertEqual(itn.normalize("リンゴはひとつじゅうごえんです"), "リンゴはひとつ15円です")
        XCTAssertEqual(itn.normalize("合計は三百円になります"), "合計は300円になります")
    }

    func testDateNormalization() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("にせんにじゅうろくねんきがつむいか"), "2026年9月6日")
        XCTAssertEqual(itn.normalize("にせんにじゅうろくねんくがつむいか"), "2026年9月6日")
        XCTAssertEqual(itn.normalize("二千二十六年九月六日"), "2026年9月6日")

        // Native day readings (1st-10th, 14th, 20th, 24th).
        XCTAssertEqual(itn.normalize("いちがつついたち"), "1月1日")
        XCTAssertEqual(itn.normalize("にがつふつか"), "2月2日")
        XCTAssertEqual(itn.normalize("さんがつみっか"), "3月3日")
        XCTAssertEqual(itn.normalize("しがつよっか"), "4月4日")
        XCTAssertEqual(itn.normalize("ごがついつか"), "5月5日")
        XCTAssertEqual(itn.normalize("ろくがつむいか"), "6月6日")
        XCTAssertEqual(itn.normalize("しちがつなのか"), "7月7日")
        XCTAssertEqual(itn.normalize("はちがつようか"), "8月8日")
        XCTAssertEqual(itn.normalize("くがつここのか"), "9月9日")
        XCTAssertEqual(itn.normalize("じゅうがつとおか"), "10月10日")
        XCTAssertEqual(itn.normalize("じゅういちがつじゅうよっか"), "11月14日")
        XCTAssertEqual(itn.normalize("じゅうにがつはつか"), "12月20日")
        XCTAssertEqual(itn.normalize("にじゅうよっか"), "24日")

        XCTAssertEqual(itn.normalize("にせんにじゅうよんねん"), "2024年")
        XCTAssertEqual(itn.normalize("二千二十五年"), "2025年")
    }

    func testPhoneNumberNormalization() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("ぜろいちにいぜろ"), "0120-")
        XCTAssertEqual(itn.normalize("ぜろさん"), "03-")

        XCTAssertEqual(itn.normalize("電話番号はぜろさんです"), "電話番号は03-です")
    }

    func testPercentageNormalization() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("ひゃくぱーせんと"), "100%")
        XCTAssertEqual(itn.normalize("ごじゅっぱーせんと"), "50%")
        XCTAssertEqual(itn.normalize("ぜろぱーせんと"), "0%")
        XCTAssertEqual(itn.normalize("にじゅうごぱーせんと"), "25%")

        XCTAssertEqual(itn.normalize("百パーセント"), "100%")
        XCTAssertEqual(itn.normalize("五十パーセント"), "50%")
        XCTAssertEqual(itn.normalize("二十五％"), "25%")
        XCTAssertEqual(itn.normalize("百%"), "100%")

        XCTAssertEqual(itn.normalize("成功率はひゃくぱーせんとです"), "成功率は100%です")
    }

    func testTimeNormalization() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("十二時三十分"), "12時30分")
        XCTAssertEqual(itn.normalize("じゅうにじさんじっぷん"), "12時30分")
        XCTAssertEqual(itn.normalize("ごじじゅうごふん"), "5時15分")
        XCTAssertEqual(itn.normalize("三時"), "3時")
        XCTAssertEqual(itn.normalize("さんじ"), "3時")
    }

    func testGeneralNumbersAndUnits() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("千五百"), "1500")
        XCTAssertEqual(itn.normalize("二万"), "20000")
        XCTAssertEqual(itn.normalize("三万五千"), "35000")
        XCTAssertEqual(itn.normalize("十五個"), "15個")
        XCTAssertEqual(itn.normalize("さんぼん"), "3本")
        XCTAssertEqual(itn.normalize("五人"), "5人")
        XCTAssertEqual(itn.normalize("十回"), "10回")

        // The honorific さん must not become 3.
        XCTAssertEqual(itn.normalize("山田さん"), "山田さん")
        XCTAssertEqual(itn.normalize("一度"), "1度")
    }

    func testEdgeCases() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize(""), "")
        XCTAssertEqual(itn.normalize("東京に行きます"), "東京に行きます")
        XCTAssertEqual(itn.normalize("本日は晴天なり"), "本日は晴天なり")
    }

    func testPositionalKanjiAndKanaDigits() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("二〇二六"), "2026")
        XCTAssertEqual(itn.normalize("二〇二六年"), "2026年")
        XCTAssertEqual(itn.normalize("一五円"), "15円")
        XCTAssertEqual(itn.normalize("一〇〇"), "100")
        XCTAssertEqual(itn.normalize("五番"), "5番")
    }

    func testColloquialGrammarProtection() {
        let itn = InverseTextNormalizer()

        // なのか, ようか, いつか must not become dates (7th, 8th, 5th).
        XCTAssertEqual(itn.normalize("本当なのか"), "本当なのか")
        XCTAssertEqual(itn.normalize("映画を見ようか"), "映画を見ようか")
        XCTAssertEqual(itn.normalize("いつかまた会おう"), "いつかまた会おう")

        // Verb phrases such as 気がつく and とお書き must not become months or days.
        XCTAssertEqual(itn.normalize("それに気がつきました"), "それに気がつきました")
        XCTAssertEqual(itn.normalize("鉛筆とお書きください"), "鉛筆とお書きください")
    }

    func testParticleNiAndDenseTokenPreservation() {
        let itn = InverseTextNormalizer()

        // The particle に after 6日 must not merge with 15円 into 25円.
        XCTAssertEqual(itn.normalize("にせんにじゅうろくねんきがつむいかにじゅうごえん"), "2026年9月6日に15円")

        // Dates, counters, and currency packed into one utterance.
        let denseInput = "にせんにじゅうろくねんきがつむいかにさんぼんのぺんをじゅうごえんでにこかった"
        let denseExpected = "2026年9月6日に3本のぺんを15円で2個かった"
        XCTAssertEqual(itn.normalize(denseInput), denseExpected)

        XCTAssertEqual(itn.normalize("10日に20人"), "10日に20人")

        // Honorific さん and ordinal counter 人目 stay as written.
        XCTAssertEqual(itn.normalize("3人目の山田さん"), "3人目の山田さん")
    }

    func testMixedArabicAndKanjiCurrencyAndNumbers() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("1万円"), "10000円")
        XCTAssertEqual(itn.normalize("2万円"), "20000円")
        XCTAssertEqual(itn.normalize("100万円"), "1000000円")
        XCTAssertEqual(itn.normalize("1万五千円"), "15000円")
        XCTAssertEqual(itn.normalize("1兆円"), "1000000000000円")
        XCTAssertEqual(itn.normalize("一兆円"), "1000000000000円")
    }

    func testHomophoneAndIdiomProtection() {
        let itn = InverseTextNormalizer()

        // Kana words that contain numeral sounds.
        XCTAssertEqual(itn.normalize("まんがを読む"), "まんがを読む")
        XCTAssertEqual(itn.normalize("おくびょうな性格"), "おくびょうな性格")
        XCTAssertEqual(itn.normalize("せんせいに相談する"), "せんせいに相談する")
        XCTAssertEqual(itn.normalize("せんたくきを回す"), "せんたくきを回す")
        XCTAssertEqual(itn.normalize("じゅうしょを記入する"), "じゅうしょを記入する")

        // Kanji idioms that contain numerals.
        XCTAssertEqual(itn.normalize("万全を期す"), "万全を期す")
        XCTAssertEqual(itn.normalize("万が一の備え"), "万が一の備え")
        XCTAssertEqual(itn.normalize("百景を楽しむ"), "百景を楽しむ")
        XCTAssertEqual(itn.normalize("千里眼を持つ"), "千里眼を持つ")
        XCTAssertEqual(itn.normalize("百科事典を調べる"), "百科事典を調べる")
        XCTAssertEqual(itn.normalize("ごはんを食べる"), "ごはんを食べる")
        XCTAssertEqual(itn.normalize("しかくを取得する"), "しかくを取得する")
        XCTAssertEqual(itn.normalize("にくを焼く"), "にくを焼く")
    }

    func testCompoundDigitsWithVariousUnits() {
        let itn = InverseTextNormalizer()

        // Compounds whose last digit is a one-mora numeral (に, し, ご, く).
        XCTAssertEqual(itn.normalize("にじゅうごさい"), "25歳")
        XCTAssertEqual(itn.normalize("さんじゅうにさい"), "32歳")
        XCTAssertEqual(itn.normalize("よんじゅうよんさい"), "44歳")
        XCTAssertEqual(itn.normalize("ごじゅうきゅうさい"), "59歳")

        // Boundary with the counters にん and にち.
        XCTAssertEqual(itn.normalize("じゅうにん"), "10人")
        XCTAssertEqual(itn.normalize("じゅうににん"), "12人")
        XCTAssertEqual(itn.normalize("じゅうにち"), "10日")
        XCTAssertEqual(itn.normalize("じゅうににち"), "12日")

        // A single one-mora numeral followed by a unit.
        XCTAssertEqual(itn.normalize("ごぱーせんと"), "5%")
        XCTAssertEqual(itn.normalize("にばい"), "2倍")
        XCTAssertEqual(itn.normalize("ごかい"), "5回")
    }
}
