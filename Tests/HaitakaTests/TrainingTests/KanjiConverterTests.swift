import XCTest
@testable import Haitaka

final class KanjiConverterTests: XCTestCase {
    func testKanaNormalization() {
        let converter = KanjiConverter()
        
        let text = "カタカナとひらがな"
        let normalized = converter.normalizeKana(text)
        XCTAssertEqual(normalized, "かたかなとひらがな")
        
        // Kanji is left unchanged.
        let kanjiText = "日本語"
        let kanjiRes = converter.normalizeKana(kanjiText)
        XCTAssertEqual(kanjiRes, "日本語")
    }

    func testEmptyAndSpecialCharacters() {
        let converter = KanjiConverter()
        
        let empty = converter.normalizeKana("")
        XCTAssertEqual(empty, "")
        XCTAssertEqual(converter.convertToHiragana(""), "")
    }

    func testNumberReadingPositional() {
        XCTAssertEqual(KanjiConverter.numberReading("0"), "ぜろ")
        XCTAssertEqual(KanjiConverter.numberReading("7"), "なな")
        XCTAssertEqual(KanjiConverter.numberReading("20"), "にじゅう")
        XCTAssertEqual(KanjiConverter.numberReading("213"), "にひゃくじゅうさん")
        XCTAssertEqual(KanjiConverter.numberReading("226"), "にひゃくにじゅうろく")
        XCTAssertEqual(KanjiConverter.numberReading("300"), "さんびゃく")
        XCTAssertEqual(KanjiConverter.numberReading("600"), "ろっぴゃく")
        XCTAssertEqual(KanjiConverter.numberReading("800"), "はっぴゃく")
        XCTAssertEqual(KanjiConverter.numberReading("3000"), "さんぜん")
        XCTAssertEqual(KanjiConverter.numberReading("8000"), "はっせん")
        XCTAssertEqual(KanjiConverter.numberReading("1473"), "せんよんひゃくななじゅうさん")
        XCTAssertEqual(KanjiConverter.numberReading("10000"), "いちまん")
        XCTAssertEqual(KanjiConverter.numberReading("100000000"), "いちおく")
        XCTAssertEqual(KanjiConverter.numberReading("12345678"), "せんにひゃくさんじゅうよんまんごせんろっぴゃくななじゅうはち")
        XCTAssertEqual(KanjiConverter.numberReading("１４７３"), "せんよんひゃくななじゅうさん")
    }

    func testNumberReadingDigitWise() {
        // A leading zero (phone numbers) switches to digit-by-digit reading.
        XCTAssertEqual(KanjiConverter.numberReading("0120"), "ぜろいちにぜろ")
        // Non-digit characters give nil.
        XCTAssertNil(KanjiConverter.numberReading("１２c"))
        XCTAssertNil(KanjiConverter.numberReading("abc"))
        XCTAssertNil(KanjiConverter.numberReading(""))
    }

    func testTokenizeIncludesNumberReadings() {
        let converter = KanjiConverter()
        XCTAssertEqual(converter.convertToHiragana("１４７３年"), "せんよんひゃくななじゅーさんねん")
        XCTAssertEqual(converter.convertToHiragana("２０億円"), "にじゅーおくえん")
    }

    // MARK: - Long-vowel mark restoration and normalization

    func testCombiningMacronRestoration() {
        let converter = KanjiConverter()

        // Long-vowel marks in katakana loanwords survive conversion.
        XCTAssertEqual(converter.convertToHiragana("マレーシア"), "まれーしあ")
        XCTAssertEqual(converter.convertToHiragana("データ"), "でーた")
        XCTAssertEqual(converter.convertToHiragana("あーあ"), "あーあ")
        XCTAssertEqual(converter.convertToHiragana("コンピューター"), "こんぴゅーたー")
        XCTAssertEqual(converter.convertToHiragana("パーティー"), "ぱーてぃー")
        XCTAssertEqual(converter.convertToHiragana("コーヒー"), "こーひー")
        XCTAssertEqual(converter.convertToHiragana("サーバー"), "さーばー")
        XCTAssertEqual(converter.convertToHiragana("ビール"), "びーる")
    }

    func testWaveDashAndProlongedNormalization() {
        let converter = KanjiConverter()

        // Wave dash 〜 and full-width tilde ～ become ー.
        XCTAssertEqual(converter.convertToHiragana("ら〜めん"), "らーめん")
        XCTAssertEqual(converter.convertToHiragana("ら～めん"), "らーめん")
        XCTAssertEqual(converter.convertToHiragana("う〜ん"), "うーん")
        XCTAssertEqual(converter.convertToHiragana("あ〜〜"), "あーー")

        // Half-width ｰ (U+FF70), horizontal bar ― (U+2015), and em dash — (U+2014) become ー.
        XCTAssertEqual(converter.convertToHiragana("らｰめん"), "らーめん")
        XCTAssertEqual(converter.convertToHiragana("あ――"), "あーー")
        XCTAssertEqual(converter.convertToHiragana("え——"), "えーー")

        // normalizeKana and kanaOnly normalize the same way.
        XCTAssertEqual(converter.normalizeKana("ら〜めん"), "らーめん")
        XCTAssertEqual(converter.normalizeKana("う～ん"), "うーん")
        XCTAssertEqual(converter.normalizeKana("らｰめん"), "らーめん")
        XCTAssertEqual(converter.kanaOnly("あ〜い～う"), "あーいーう")
        XCTAssertEqual(converter.kanaOnly("らｰめん"), "らーめん")
        XCTAssertEqual(converter.kanaOnly("あ――"), "あーー")
    }

    func testPronunciationNormalization() {
        let converter = KanjiConverter()

        // Repeated vowels become a long-vowel mark (spelling follows sound).
        XCTAssertEqual(converter.convertToHiragana("東京"), "とーきょー")
        XCTAssertEqual(converter.convertToHiragana("先生"), "せんせー")
        XCTAssertEqual(converter.convertToHiragana("映画"), "えーが")
        XCTAssertEqual(converter.convertToHiragana("大きい"), "おーきー")
        XCTAssertEqual(converter.convertToHiragana("おじいさん"), "おじーさん")
        XCTAssertEqual(converter.convertToHiragana("おばあさん"), "おばーさん")
        XCTAssertEqual(converter.convertToHiragana("氷"), "こーり")
        XCTAssertEqual(converter.convertToHiragana("空気"), "くーき")
        // Verb-final う stays a vowel.
        XCTAssertEqual(converter.convertToHiragana("思う"), "おもう")
        XCTAssertEqual(converter.convertToHiragana("買う"), "かう")
        // Particles and historical kana spelling.
        XCTAssertEqual(converter.convertToHiragana("これは水を飲む"), "これわみずおのむ")
        XCTAssertEqual(converter.convertToHiragana("東京へ行く"), "とーきょーえいく")
        XCTAssertEqual(converter.convertToHiragana("続く"), "つずく")
        // を, ぢ, and づ never appear in the vocabulary.
        XCTAssertFalse(converter.convertToHiragana("鼻血を出す").contains("ぢ"))
    }

    func testAlphabetTokenProtectionAndBilingualSeed() {
        let converter = KanjiConverter()

        // Common tech terms and loanwords get proper kana readings, not letter-by-letter ones (いぷほね, あっぷれ).
        let tokens = converter.tokenize("iPhone")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens.first?.surface, "iPhone")
        XCTAssertEqual(tokens.first?.reading, "あいふぉーん")

        let appleTokens = converter.tokenize("Apple")
        XCTAssertEqual(appleTokens.count, 1)
        XCTAssertEqual(appleTokens.first?.surface, "Apple")
        XCTAssertEqual(appleTokens.first?.reading, "あっぷる")

        let googleTokens = converter.tokenize("Google")
        XCTAssertEqual(googleTokens.first?.reading, "ぐーぐる")

        let aiTokens = converter.tokenize("AI")
        XCTAssertEqual(aiTokens.first?.reading, "えーあい")

        let pcTokens = converter.tokenize("PC")
        XCTAssertEqual(pcTokens.first?.reading, "ぴーしー")

        let wifiTokens = converter.tokenize("WiFi")
        XCTAssertEqual(wifiTokens.first?.reading, "わいふぁい")

        let wifiHyphenTokens = converter.tokenize("Wi-Fi")
        XCTAssertEqual(wifiHyphenTokens.first?.surface, "Wi-Fi")
        XCTAssertEqual(wifiHyphenTokens.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi-Fi"), "わいふぁい")

        let openAiTokens = converter.tokenize("OpenAI")
        XCTAssertEqual(openAiTokens.first?.reading, "おーぷんえーあい")

        let airbnbTokens = converter.tokenize("Airbnb")
        XCTAssertEqual(airbnbTokens.first?.reading, "えあびー")

        // Uppercase acronyms are spelled out letter by letter.
        let gpuTokens = converter.tokenize("GPU")
        XCTAssertEqual(gpuTokens.first?.reading, "じーぴーゆー")

        let sdkTokens = converter.tokenize("SDK")
        XCTAssertEqual(sdkTokens.first?.reading, "えすでぃーけー")

        // Unknown English words get an empty reading instead of a broken romaji reading.
        let unknownTokens = converter.tokenize("unknownfoobar")
        XCTAssertEqual(unknownTokens.count, 1)
        XCTAssertEqual(unknownTokens.first?.reading, "")

        let mixed = converter.convertToHiragana("iPhoneとMacBookを使って作業する")
        XCTAssertEqual(mixed, "あいふぉーんとまっくぶっくおつかってさぎょーする")
    }

    // MARK: - Edge cases and unusual input

    func testPhase2EdgeCasesAndExtremeInputs() {
        let converter = KanjiConverter()

        XCTAssertTrue(converter.tokenize("").isEmpty)
        XCTAssertEqual(converter.normalizeKana(""), "")
        XCTAssertEqual(converter.convertToHiragana(""), "")

        // Wave dashes and prolonged marks only.
        XCTAssertEqual(converter.convertToHiragana("〜〜〜"), "ーーー")
        XCTAssertEqual(converter.convertToHiragana("あ〜〜〜〜"), "あーーーー")

        // Full-width Latin letters.
        let fullwidthTokens = converter.tokenize("ｉＰｈｏｎｅ")
        XCTAssertEqual(fullwidthTokens.first?.reading, "あいふぉーん")

        // Case variations.
        let upperIPhone = converter.tokenize("IPHONE")
        XCTAssertEqual(upperIPhone.first?.reading, "あいふぉーん")

        let lowerApple = converter.tokenize("apple")
        XCTAssertEqual(lowerApple.first?.reading, "あっぷる")

        // Single letters.
        let singleA = converter.tokenize("A")
        XCTAssertEqual(singleA.first?.reading, "えー")
        let singleZ = converter.tokenize("Z")
        XCTAssertEqual(singleZ.first?.reading, "ぜっと")

        // Unknown uppercase acronym.
        let unknownAcronym = converter.tokenize("XYZ")
        XCTAssertEqual(unknownAcronym.first?.reading, "えっくすわいぜっと")

        // Digits, English, long-vowel marks, and wave dashes combined.
        let complexText = "iPhone 15の価格は１４万円〜１５万円です。"
        let hira = converter.convertToHiragana(complexText)
        XCTAssertTrue(hira.contains("あいふぉーん"))
        XCTAssertTrue(hira.contains("じゅーご"))
        XCTAssertTrue(hira.contains("ー"))

        // Wi-Fi with full-width hyphens and various dashes.
        let fullwidthWifi = converter.tokenize("Ｗｉ－Ｆｉ")
        XCTAssertEqual(fullwidthWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Ｗｉ－Ｆｉ"), "わいふぁい")

        let enDashWifi = converter.tokenize("Wi–Fi")
        XCTAssertEqual(enDashWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi–Fi"), "わいふぁい")

        let emDashWifi = converter.tokenize("Wi—Fi")
        XCTAssertEqual(emDashWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi—Fi"), "わいふぁい")

        let barWifi = converter.tokenize("Wi―Fi")
        XCTAssertEqual(barWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi―Fi"), "わいふぁい")

        // Mirrativ reading derived from the seed vocabulary.
        let mirrativTokens = converter.tokenize("Mirrativ")
        XCTAssertEqual(mirrativTokens.first?.reading, "みらてぃぶ")
        XCTAssertEqual(converter.convertToHiragana("Mirrativ"), "みらてぃぶ")

        // B-to-B and hyphenated acronyms.
        let btobTokens = converter.tokenize("B-to-B")
        XCTAssertEqual(btobTokens.first?.reading, "びーとぅーびー")
        XCTAssertEqual(converter.convertToHiragana("B-to-B"), "びーとぅーびー")

        let hyphenAcronym = converter.tokenize("Ｘ－Ｙ－Ｚ")
        XCTAssertEqual(hyphenAcronym.first?.reading, "えっくすわいぜっと")
    }

    func testReadingOverrides() {
        let converter = KanjiConverter()
        // Morphological analyzer readings are adjusted to spoken forms.
        XCTAssertEqual(converter.convertToHiragana("私は皆と日本へ行く"), "わたしわみんなとにほんえいく")
        XCTAssertEqual(converter.convertToHiragana("皆さん、明日と一昨日"), "みなさんあしたとおととい")
        XCTAssertEqual(converter.convertToHiragana("そういうことだと言う"), "そーゆーことだとゆー")
        XCTAssertEqual(converter.convertToHiragana("言いました"), "いーました")
        // Compound words are merged.
        XCTAssertEqual(converter.convertToHiragana("お母さんとお父さんとお兄さんとお姉さん"), "おかーさんとおとーさんとおにーさんとおねーさん")
        XCTAssertEqual(converter.convertToHiragana("木曜日と日曜日"), "もくよーびとにちよーび")
        XCTAssertEqual(converter.convertToHiragana("世界中"), "せかいじゅー")
        XCTAssertEqual(converter.tokenize("お母さん").count, 1)
        // 人 after a country name reads じん; after a numeral it reads にん.
        XCTAssertEqual(converter.convertToHiragana("日本人とフランス人と外国人"), "にほんじんとふらんすじんとがいこくじん")
        XCTAssertEqual(converter.convertToHiragana("3人と何人"), "さんにんとなんにん")
        // 何 reads なに before a particle or verb, なん before a counter or で.
        XCTAssertEqual(converter.convertToHiragana("何が何を何してる"), "なにがなにおなにしてる")
        XCTAssertEqual(converter.convertToHiragana("何で何回"), "なんでなんかい")
        XCTAssertEqual(converter.convertToHiragana("何か"), "なにか")
        // 良い and 得る, fixed readings (下/もと -> した), and readings before particles (他の -> ほかの, 数は -> かずは).
        XCTAssertEqual(converter.convertToHiragana("良い天気"), "いーてんき")
        XCTAssertEqual(converter.convertToHiragana("雲の下で"), "くものしたで")
        XCTAssertEqual(converter.convertToHiragana("米を炊く"), "こめおたく")
        XCTAssertEqual(converter.convertToHiragana("他の人と他に"), "ほかのひととほかに")
        XCTAssertEqual(converter.convertToHiragana("その数は"), "そのかずわ")
        XCTAssertEqual(converter.convertToHiragana("数か月"), "すーかげつ")
        // Rendaku after a noun; unchanged after words ending in kana (この, 深い).
        XCTAssertEqual(converter.convertToHiragana("ビル作りと多摩川"), "びるずくりとたまがわ")
        XCTAssertEqual(converter.convertToHiragana("ゲーム好きと腕時計と十時頃"), "げーむずきとうでどけーとじゅーじごろ")
        XCTAssertEqual(converter.convertToHiragana("この通り、深い川"), "このとーりふかいかわ")
        XCTAssertEqual(converter.convertToHiragana("兄ちゃんと姉さん"), "にーちゃんとねーさん")
    }

    func testCounterSandhi() {
        let converter = KanjiConverter()
        // Gemination and p-sound changes (same for Arabic and kanji numerals).
        XCTAssertEqual(converter.convertToHiragana("1本10本100回1週間"), "いっぽんじゅっぽんひゃっかいいっしゅーかん")
        XCTAssertEqual(converter.convertToHiragana("一杯六杯八匹十分"), "いっぱいろっぱいはっぴきじゅっぷん")
        XCTAssertEqual(converter.convertToHiragana("1か月1キロ10ページ"), "いっかげついっきろじゅっぺーじ")
        // After ん, counters voice (本, 杯, 匹) or take the p-sound (分, 歩, 発).
        XCTAssertEqual(converter.convertToHiragana("3本三杯何本千本3分三歩"), "さんぼんさんばいなんぼんせんぼんさんぷんさんぽ")
        // 4, 7, 9, 2, and 5 do not change (also fixes the analyzer returning よんぽん for 四本).
        XCTAssertEqual(converter.convertToHiragana("4本7本9本2分4分"), "よんほんななほんきゅーほんにふんよんぷん")
        // 6 and 100 geminate only before k- and h-row counters.
        XCTAssertEqual(converter.convertToHiragana("六冊百冊六回"), "ろくさつひゃくさつろっかい")
        // h-row loanword units do not geminate.
        XCTAssertEqual(converter.convertToHiragana("1ヘクタール1パーセント"), "いちへくたーるいっぱーせんと")
        // Counters whose reading depends on the number.
        XCTAssertEqual(converter.convertToHiragana("4月4日4時"), "しがつよっかよじ")
        XCTAssertEqual(converter.convertToHiragana("9月10日7時"), "くがつとーかしちじ")
        XCTAssertEqual(converter.convertToHiragana("11日20日24日"), "じゅーいちにちはつかにじゅーよっか")
        XCTAssertEqual(converter.convertToHiragana("1人2人4人3人"), "ひとりふたりよにんさんにん")
        XCTAssertEqual(converter.convertToHiragana("1つ3つ四つ八つ"), "ひとつみっつよっつやっつ")
        // 位置 is not a numeral, so it does not geminate.
        XCTAssertEqual(converter.convertToHiragana("位置確認"), "いちかくにん")
    }

    func testTranscriptionMarkersAreSilent() {
        let converter = KanjiConverter()
        // (笑), (爆笑), and (冷笑) mark laughter and are not pronounced.
        XCTAssertEqual(converter.convertToHiragana("(笑)ん?撮ってない?"), "んとってない")
        XCTAssertEqual(converter.convertToHiragana("やられた(爆笑)駆除した（冷笑）ね"), "やられたくじょしたね")
        // Parenthesized text without 笑 is kept.
        XCTAssertEqual(converter.convertToHiragana("彼(兄)が来た"), "かれあにがきた")
        // A standalone w is read aloud as だぶりゅー, so it is kept.
        XCTAssertEqual(converter.convertToHiragana("それなw"), "それなだぶりゅー")
        XCTAssertEqual(converter.convertToHiragana("Wi-Fiとweb"), "わいふぁいとうぇぶ")
    }

    func testEnglishWordsAndKimi() {
        let converter = KanjiConverter()
        // Unknown English words get an empty reading, even with punctuation; uppercase words of 5+ letters are not spelled out.
        XCTAssertEqual(converter.convertToHiragana("Have a sweet dream.See you"), "えー")
        XCTAssertEqual(converter.convertToHiragana("Thank you.今日がね"), "きょーがね")
        XCTAssertEqual(converter.convertToHiragana("ZENSHOグループ"), "ぐるーぷ")
        XCTAssertEqual(converter.convertToHiragana("GPU"), "じーぴーゆー")
        // 君 reads きみ at sentence start or after a particle, くん after a name.
        XCTAssertEqual(converter.convertToHiragana("君を思い出すよ"), "きみおおもいだすよ")
        XCTAssertEqual(converter.convertToHiragana("でも君は"), "でもきみわ")
        XCTAssertEqual(converter.convertToHiragana("たかし君が来た"), "たかしくんがきた")
    }
}
