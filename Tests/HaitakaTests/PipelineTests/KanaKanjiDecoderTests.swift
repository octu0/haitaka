import XCTest
@testable import Haitaka

final class KanaKanjiDecoderTests: XCTestCase {

    func testSeedVocabularyContents() {
        let entries = SeedVocabulary.entries
        XCTAssertLessThanOrEqual(50, entries.count)

        // Abbreviations and nicknames.
        let subaraEntries = entries.filter { $0.reading == "すたば" && $0.surface == "スターバックス" }
        XCTAssertFalse(subaraEntries.isEmpty)

        let macdoEntries = entries.filter { $0.reading == "まくど" && $0.surface == "マクドナルド" }
        XCTAssertFalse(macdoEntries.isEmpty)

        let mercariEntries = entries.filter { $0.reading == "めすか" && $0.surface == "メルカリ" }
        XCTAssertFalse(mercariEntries.isEmpty)

        let airbnbEntries = entries.filter { $0.reading == "えあびー" && $0.surface == "Airbnb" }
        XCTAssertFalse(airbnbEntries.isEmpty)

        // Brand names with Latin-script surfaces.
        let iphoneEntries = entries.filter { $0.reading == "あいふぉーん" && $0.surface == "iPhone" }
        XCTAssertFalse(iphoneEntries.isEmpty)

        let appleEntriesEn = entries.filter { $0.reading == "あっぷる" && $0.surface == "Apple" }
        let appleEntriesJa = entries.filter { $0.reading == "あっぷる" && $0.surface == "アップル" }
        XCTAssertFalse(appleEntriesEn.isEmpty)
        XCTAssertFalse(appleEntriesJa.isEmpty)

        let googleEntries = entries.filter { $0.reading == "ぐーぐる" && $0.surface == "Google" }
        XCTAssertFalse(googleEntries.isEmpty)

        let openAiEntries = entries.filter { $0.reading == "おーぷんえーあい" && $0.surface == "OpenAI" }
        XCTAssertFalse(openAiEntries.isEmpty)

        let tiktokEntries = entries.filter { $0.reading == "てぃっくとっく" && $0.surface == "TikTok" }
        XCTAssertFalse(tiktokEntries.isEmpty)

        let netflixEntries = entries.filter { $0.reading == "ねっとふりっくす" && $0.surface == "Netflix" }
        XCTAssertFalse(netflixEntries.isEmpty)

        let youtubeEntries = entries.filter { $0.reading == "ゆーちゅーぶ" && $0.surface == "YouTube" }
        XCTAssertFalse(youtubeEntries.isEmpty)
    }

    func testKanaKanjiDictionarySeedIntegration() {
        // The default initializer loads the seed vocabulary.
        let dict = KanaKanjiDictionary()
        XCTAssertLessThanOrEqual(50, dict.count)

        let iphoneHits = dict.lookupExact(reading: "あいふぉーん")
        XCTAssertFalse(iphoneHits.isEmpty)
        XCTAssertEqual(iphoneHits.first?.surface, "iPhone")

        let starbucksHits = dict.lookupExact(reading: "すたば")
        XCTAssertFalse(starbucksHits.isEmpty)
        XCTAssertEqual(starbucksHits.first?.surface, "スターバックス")

        let macdoHits = dict.lookupExact(reading: "まくど")
        XCTAssertFalse(macdoHits.isEmpty)
        XCTAssertEqual(macdoHits.first?.surface, "マクドナルド")

        let mercariHits = dict.lookupExact(reading: "めすか")
        XCTAssertFalse(mercariHits.isEmpty)
        XCTAssertEqual(mercariHits.first?.surface, "メルカリ")

        let airbnbHits = dict.lookupExact(reading: "えあびー")
        XCTAssertFalse(airbnbHits.isEmpty)
        XCTAssertEqual(airbnbHits.first?.surface, "Airbnb")

        let openAiHits = dict.lookupExact(reading: "おーぷんえーあい")
        XCTAssertFalse(openAiHits.isEmpty)
        XCTAssertEqual(openAiHits.first?.surface, "OpenAI")

        let emptyDict = KanaKanjiDictionary(includeSeed: false)
        XCTAssertEqual(emptyDict.count, 0)
    }

    func testKanaKanjiDecoderSeedDecoding() {
        let dict = KanaKanjiDictionary()
        let decoder = KanaKanjiDecoder(dictionary: dict)

        // Abbreviations expand to full or brand names.
        let resStarbucks = decoder.decode(kanaText: "すたば")
        XCTAssertEqual(resStarbucks, "スターバックス")

        let resMacdo = decoder.decode(kanaText: "まくど")
        XCTAssertEqual(resMacdo, "マクドナルド")

        let resMercari = decoder.decode(kanaText: "めすか")
        XCTAssertEqual(resMercari, "メルカリ")

        let resAirbnb = decoder.decode(kanaText: "えあびー")
        XCTAssertEqual(resAirbnb, "Airbnb")

        // English tech proper nouns.
        let resIPhone = decoder.decode(kanaText: "あいふぉーん")
        XCTAssertEqual(resIPhone, "iPhone")

        let resOpenAI = decoder.decode(kanaText: "おーぷんえーあい")
        XCTAssertEqual(resOpenAI, "OpenAI")

        let resTikTok = decoder.decode(kanaText: "てぃっくとっく")
        XCTAssertTrue(resTikTok == "TikTok" || resTikTok == "ティックトック")

        let resYouTube = decoder.decode(kanaText: "ゆーちゅーぶ")
        XCTAssertTrue(resYouTube == "YouTube" || resYouTube == "ユーチューブ")

        let resNetflix = decoder.decode(kanaText: "ねっとふりっくす")
        XCTAssertTrue(resNetflix == "Netflix" || resNetflix == "ネットフリックス")
    }

    func testBuildFromCorpusRetainsSeedVocabulary() {
        let dict = KanaKanjiDictionary()
        let corpus = [
            "人工知能の技術が進歩している。",
            "今日は良い天気ですね。"
        ]
        dict.buildFromCorpus(rawTexts: corpus)

        // Corpus words are indexed by pronunciation kana (おう -> おー).
        let jinkouHits = dict.lookupExact(reading: "じんこー")
        XCTAssertFalse(jinkouHits.isEmpty)
        XCTAssertEqual(jinkouHits.first?.surface, "人工")

        // Alternative spoken readings map back to the same surface.
        dict.buildFromCorpus(rawTexts: ["私は皆と行く。"])
        XCTAssertEqual(dict.lookupExact(reading: "わたし").first?.surface, "私")
        XCTAssertEqual(dict.lookupExact(reading: "あたし").first?.surface, "私")
        XCTAssertEqual(dict.lookupExact(reading: "わたくし").first?.surface, "私")
        XCTAssertEqual(dict.lookupExact(reading: "みんな").first?.surface, "皆")
        XCTAssertEqual(dict.lookupExact(reading: "みな").first?.surface, "皆")

        let tenkiHits = dict.lookupExact(reading: "てんき")
        XCTAssertFalse(tenkiHits.isEmpty)
        XCTAssertEqual(tenkiHits.first?.surface, "天気")

        // Seed entries survive corpus building.
        let iphoneHits = dict.lookupExact(reading: "あいふぉーん")
        XCTAssertFalse(iphoneHits.isEmpty)
        XCTAssertEqual(iphoneHits.first?.surface, "iPhone")

        let starbucksHits = dict.lookupExact(reading: "すたば")
        XCTAssertFalse(starbucksHits.isEmpty)
        XCTAssertEqual(starbucksHits.first?.surface, "スターバックス")
    }

    func testFuzzyPhoneticSearchWithSeedVocabulary() {
        let dict = KanaKanjiDictionary()

        // まくと (unvoiced variant of まくど) -> マクドナルド
        let fuzzyMacdo = dict.lookupFuzzyPhonetic(reading: "まくと", maxPhoneticDist: 0.5)
        let foundMacdo = fuzzyMacdo.contains { $0.entry.surface == "マクドナルド" }
        XCTAssertTrue(foundMacdo)

        // すたぱ (semi-voiced variant of すたば) -> スターバックス
        let fuzzyStarbucks = dict.lookupFuzzyPhonetic(reading: "すたぱ", maxPhoneticDist: 0.5)
        let foundStarbucks = fuzzyStarbucks.contains { $0.entry.surface == "スターバックス" }
        XCTAssertTrue(foundStarbucks)
    }

    func testPhase2DecoderEdgeCases() {
        let dict = KanaKanjiDictionary()
        let decoder = KanaKanjiDecoder(dictionary: dict)

        XCTAssertEqual(decoder.decode(kanaText: ""), "")
        XCTAssertTrue(decoder.decodeNBest(kanaText: "", beamWidth: 3).isEmpty)

        // Unknown kana passes through unchanged.
        let unkResult = decoder.decode(kanaText: "んんん")
        XCTAssertEqual(unkResult, "んんん")

        let emptyDict = KanaKanjiDictionary(includeSeed: false)
        let emptyDecoder = KanaKanjiDecoder(dictionary: emptyDict)
        let emptyRes = emptyDecoder.decode(kanaText: "あいうえお")
        XCTAssertEqual(emptyRes, "あいうえお")

        // Alphabet lookups fold case and full-width forms.
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("iphone"), "あいふぉーん")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("IPHONE"), "あいふぉーん")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("ｉＰｈｏｎｅ"), "あいふぉーん")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("mirrativ"), "みらてぃぶ")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("MIRRATIV"), "みらてぃぶ")
        XCTAssertEqual(SeedVocabulary.readingForLetter("a"), "えー")
        XCTAssertEqual(SeedVocabulary.readingForLetter("Ａ"), "えー")
        XCTAssertNil(SeedVocabulary.readingForAlphabetSurface(""))
        XCTAssertNil(SeedVocabulary.readingForLetter(""))

        // Long-vowel marks alone or repeated pass through unchanged.
        XCTAssertEqual(decoder.decode(kanaText: "ー"), "ー")
        XCTAssertEqual(decoder.decode(kanaText: "ーーーーー"), "ーーーーー")
        XCTAssertEqual(decoder.decode(kanaText: "あああーーーーー"), "あああーーーーー")

        // Proper nouns inside a longer kana stream.
        let mixedStream = decoder.decode(kanaText: "えあびーでとまったあとめるかりでうった")
        XCTAssertTrue(mixedStream.contains("Airbnb"))
        XCTAssertTrue(mixedStream.contains("メルカリ"))

        XCTAssertEqual(decoder.decode(kanaText: "びーとぅーびー"), "BtoB")
    }
}
