import Foundation

/// Built-in readings and surfaces for proper nouns and loanwords. The same table serves label normalization
/// (surface -> kana for training text) and kana-kanji conversion (kana -> surface).
public struct SeedVocabulary: Sendable {
    public static let entries: [KanaKanjiEntry] = [
        // MARK: - Colloquial names and abbreviations
        KanaKanjiEntry(reading: "えあびー", surface: "Airbnb", frequency: 100),
        KanaKanjiEntry(reading: "えあびーあんどびー", surface: "Airbnb", frequency: 100),
        KanaKanjiEntry(reading: "えあびーえぬびー", surface: "Airbnb", frequency: 100),
        KanaKanjiEntry(reading: "めすか", surface: "メルカリ", frequency: 100),
        KanaKanjiEntry(reading: "めるかり", surface: "メルカリ", frequency: 100),
        KanaKanjiEntry(reading: "めるかり", surface: "Mercari", frequency: 80),
        KanaKanjiEntry(reading: "みらてぃぶ", surface: "ミラティブ", frequency: 100),
        KanaKanjiEntry(reading: "みらてぃぶ", surface: "Mirrativ", frequency: 100),
        KanaKanjiEntry(reading: "すたば", surface: "スターバックス", frequency: 100),
        KanaKanjiEntry(reading: "すたーばっくす", surface: "スターバックス", frequency: 100),
        KanaKanjiEntry(reading: "まくど", surface: "マクドナルド", frequency: 100),
        KanaKanjiEntry(reading: "まくどなるど", surface: "マクドナルド", frequency: 100),
        KanaKanjiEntry(reading: "こんびに", surface: "コンビニ", frequency: 100),
        KanaKanjiEntry(reading: "ふぁみま", surface: "ファミリーマート", frequency: 100),
        KanaKanjiEntry(reading: "ふぁみりーまーと", surface: "ファミリーマート", frequency: 100),
        KanaKanjiEntry(reading: "ろーそん", surface: "ローソン", frequency: 100),
        KanaKanjiEntry(reading: "せぶん", surface: "セブンイレブン", frequency: 100),
        KanaKanjiEntry(reading: "せぶんいれぶん", surface: "セブンイレブン", frequency: 100),
        KanaKanjiEntry(reading: "ぽけもん", surface: "ポケモン", frequency: 100),
        KanaKanjiEntry(reading: "すまほ", surface: "スマホ", frequency: 100),
        KanaKanjiEntry(reading: "ぱそこん", surface: "パソコン", frequency: 100),
        KanaKanjiEntry(reading: "てれわーく", surface: "テレワーク", frequency: 100),
        KanaKanjiEntry(reading: "りもーと", surface: "リモート", frequency: 100),
        KanaKanjiEntry(reading: "あぷり", surface: "アプリ", frequency: 100),
        KanaKanjiEntry(reading: "ねとふり", surface: "Netflix", frequency: 80),
        KanaKanjiEntry(reading: "ねとふり", surface: "ネットフリックス", frequency: 100),
        KanaKanjiEntry(reading: "いんすた", surface: "Instagram", frequency: 80),
        KanaKanjiEntry(reading: "いんすた", surface: "インスタ", frequency: 100),

        // MARK: - Japanese and English spellings
        KanaKanjiEntry(reading: "あっぷる", surface: "Apple", frequency: 80),
        KanaKanjiEntry(reading: "あっぷる", surface: "アップル", frequency: 100),
        KanaKanjiEntry(reading: "ぐーぐる", surface: "Google", frequency: 100),
        KanaKanjiEntry(reading: "ぐーぐる", surface: "グーグル", frequency: 100),
        KanaKanjiEntry(reading: "いんたーねっと", surface: "Internet", frequency: 100),
        KanaKanjiEntry(reading: "いんたーねっと", surface: "インターネット", frequency: 100),
        KanaKanjiEntry(reading: "あいふぉーん", surface: "iPhone", frequency: 100),
        KanaKanjiEntry(reading: "あいふぉん", surface: "iPhone", frequency: 100),
        KanaKanjiEntry(reading: "うぇぶ", surface: "Web", frequency: 100),
        KanaKanjiEntry(reading: "うぇぶ", surface: "ウェブ", frequency: 100),
        KanaKanjiEntry(reading: "えーあい", surface: "AI", frequency: 100),
        KanaKanjiEntry(reading: "ぴーしー", surface: "PC", frequency: 100),
        KanaKanjiEntry(reading: "わいふぁい", surface: "WiFi", frequency: 100),
        KanaKanjiEntry(reading: "わいふぁい", surface: "Wi-Fi", frequency: 100),
        KanaKanjiEntry(reading: "てぃっくとっく", surface: "TikTok", frequency: 100),
        KanaKanjiEntry(reading: "てぃっくとっく", surface: "ティックトック", frequency: 100),
        KanaKanjiEntry(reading: "ゆーちゅーぶ", surface: "YouTube", frequency: 100),
        KanaKanjiEntry(reading: "ゆーちゅーぶ", surface: "ユーチューブ", frequency: 100),
        KanaKanjiEntry(reading: "ねっとふりっくす", surface: "Netflix", frequency: 100),
        KanaKanjiEntry(reading: "ねっとふりっくす", surface: "ネットフリックス", frequency: 100),
        KanaKanjiEntry(reading: "おーぷんえーあい", surface: "OpenAI", frequency: 100),
        KanaKanjiEntry(reading: "まいくろそふと", surface: "Microsoft", frequency: 100),
        KanaKanjiEntry(reading: "まいくろそふと", surface: "マイクロソフト", frequency: 100),
        KanaKanjiEntry(reading: "あまぞん", surface: "Amazon", frequency: 100),
        KanaKanjiEntry(reading: "あまぞん", surface: "アマゾン", frequency: 100),
        KanaKanjiEntry(reading: "めた", surface: "Meta", frequency: 100),
        KanaKanjiEntry(reading: "めた", surface: "メタ", frequency: 100),
        KanaKanjiEntry(reading: "ふぇいすぶっく", surface: "Facebook", frequency: 100),
        KanaKanjiEntry(reading: "ふぇいすぶっく", surface: "フェイスブック", frequency: 100),
        KanaKanjiEntry(reading: "いんすたぐらむ", surface: "Instagram", frequency: 100),
        KanaKanjiEntry(reading: "いんすたぐらむ", surface: "インスタグラム", frequency: 100),
        KanaKanjiEntry(reading: "ついったー", surface: "Twitter", frequency: 100),
        KanaKanjiEntry(reading: "ついったー", surface: "ツイッター", frequency: 100),
        KanaKanjiEntry(reading: "うーばー", surface: "Uber", frequency: 100),
        KanaKanjiEntry(reading: "うーばー", surface: "ウーバー", frequency: 100),
        KanaKanjiEntry(reading: "ずーむ", surface: "Zoom", frequency: 100),
        KanaKanjiEntry(reading: "ずーむ", surface: "ズーム", frequency: 100),
        KanaKanjiEntry(reading: "すらっく", surface: "Slack", frequency: 100),
        KanaKanjiEntry(reading: "すらっく", surface: "スラック", frequency: 100),
        KanaKanjiEntry(reading: "らいん", surface: "LINE", frequency: 100),
        KanaKanjiEntry(reading: "らいん", surface: "ライン", frequency: 100),
        KanaKanjiEntry(reading: "やふー", surface: "Yahoo", frequency: 100),
        KanaKanjiEntry(reading: "やふー", surface: "ヤフー", frequency: 100),
        KanaKanjiEntry(reading: "ぎっとはぶ", surface: "GitHub", frequency: 100),
        KanaKanjiEntry(reading: "ぎっと", surface: "Git", frequency: 100),
        KanaKanjiEntry(reading: "のーしょん", surface: "Notion", frequency: 100),
        KanaKanjiEntry(reading: "どっかー", surface: "Docker", frequency: 100),
        KanaKanjiEntry(reading: "あいぱっど", surface: "iPad", frequency: 100),
        KanaKanjiEntry(reading: "まっく", surface: "Mac", frequency: 100),
        KanaKanjiEntry(reading: "まっくぶっく", surface: "MacBook", frequency: 100),
        KanaKanjiEntry(reading: "あんどろいど", surface: "Android", frequency: 100),
        KanaKanjiEntry(reading: "ういんどーず", surface: "Windows", frequency: 100),
        KanaKanjiEntry(reading: "りなっくす", surface: "Linux", frequency: 100),
        KanaKanjiEntry(reading: "ぶるーとぅーす", surface: "Bluetooth", frequency: 100),
        KanaKanjiEntry(reading: "しーぴーゆー", surface: "CPU", frequency: 100),
        KanaKanjiEntry(reading: "じーぴーゆー", surface: "GPU", frequency: 100),
        KanaKanjiEntry(reading: "えーぴーあい", surface: "API", frequency: 100),
        KanaKanjiEntry(reading: "ゆーあい", surface: "UI", frequency: 100),
        KanaKanjiEntry(reading: "ゆーえっくす", surface: "UX", frequency: 100),
        KanaKanjiEntry(reading: "ゆーあーるえる", surface: "URL", frequency: 100),
        KanaKanjiEntry(reading: "おーえす", surface: "OS", frequency: 100),
        KanaKanjiEntry(reading: "あいおーえす", surface: "iOS", frequency: 100),
        KanaKanjiEntry(reading: "あいてぃー", surface: "IT", frequency: 100),
        KanaKanjiEntry(reading: "でぃーえっくす", surface: "DX", frequency: 100),
        KanaKanjiEntry(reading: "えすえぬえす", surface: "SNS", frequency: 100),
        KanaKanjiEntry(reading: "ぴーあーる", surface: "PR", frequency: 100),
        KanaKanjiEntry(reading: "しーいーおー", surface: "CEO", frequency: 100),
        KanaKanjiEntry(reading: "しーてぃーおー", surface: "CTO", frequency: 100),
        KanaKanjiEntry(reading: "さーす", surface: "SaaS", frequency: 100),
        KanaKanjiEntry(reading: "びーとぅーびー", surface: "BtoB", frequency: 100),
        KanaKanjiEntry(reading: "びーとぅーびー", surface: "B-to-B", frequency: 80),
        KanaKanjiEntry(reading: "びーとぅーしー", surface: "BtoC", frequency: 100),
        KanaKanjiEntry(reading: "びーとぅーしー", surface: "B-to-C", frequency: 80),
        KanaKanjiEntry(reading: "すたーばっくす", surface: "Starbucks", frequency: 80),
        KanaKanjiEntry(reading: "まくどなるど", surface: "McDonald", frequency: 80),
        KanaKanjiEntry(reading: "まくどなるど", surface: "McDonalds", frequency: 80),
        KanaKanjiEntry(reading: "えむえる", surface: "ML", frequency: 100),
        KanaKanjiEntry(reading: "でぃーえる", surface: "DL", frequency: 100),
        KanaKanjiEntry(reading: "えるえるえむ", surface: "LLM", frequency: 100),
        KanaKanjiEntry(reading: "じーぴーてぃー", surface: "GPT", frequency: 100),
        KanaKanjiEntry(reading: "ちゃっとじーぴーてぃー", surface: "ChatGPT", frequency: 100),
        KanaKanjiEntry(reading: "ゆーえすびー", surface: "USB", frequency: 100),
        KanaKanjiEntry(reading: "えすえすでぃー", surface: "SSD", frequency: 100),
        KanaKanjiEntry(reading: "えいちでぃーでぃー", surface: "HDD", frequency: 100),
        KanaKanjiEntry(reading: "らむ", surface: "RAM", frequency: 100),
        KanaKanjiEntry(reading: "ろむ", surface: "ROM", frequency: 100),
        KanaKanjiEntry(reading: "えすきゅーえる", surface: "SQL", frequency: 100),
        KanaKanjiEntry(reading: "えいちてぃーえむえる", surface: "HTML", frequency: 100),
        KanaKanjiEntry(reading: "しーえすえす", surface: "CSS", frequency: 100),
        KanaKanjiEntry(reading: "すうぃふと", surface: "Swift", frequency: 100),
        KanaKanjiEntry(reading: "ぱいそん", surface: "Python", frequency: 100),
        KanaKanjiEntry(reading: "らすと", surface: "Rust", frequency: 100),
        KanaKanjiEntry(reading: "ごー", surface: "Go", frequency: 80),
        KanaKanjiEntry(reading: "じゃば", surface: "Java", frequency: 100),
        KanaKanjiEntry(reading: "たいぷすくりぷと", surface: "TypeScript", frequency: 100),
        KanaKanjiEntry(reading: "じゃばすくりぷと", surface: "JavaScript", frequency: 100),

        // MARK: - Single letters ("えー" -> "A")
        KanaKanjiEntry(reading: "えー", surface: "A", frequency: 50),
        KanaKanjiEntry(reading: "びー", surface: "B", frequency: 50),
        KanaKanjiEntry(reading: "しー", surface: "C", frequency: 50),
        KanaKanjiEntry(reading: "でぃー", surface: "D", frequency: 50),
        KanaKanjiEntry(reading: "いー", surface: "E", frequency: 50),
        KanaKanjiEntry(reading: "えふ", surface: "F", frequency: 50),
        KanaKanjiEntry(reading: "じー", surface: "G", frequency: 50),
        KanaKanjiEntry(reading: "えいち", surface: "H", frequency: 50),
        KanaKanjiEntry(reading: "あい", surface: "I", frequency: 50),
        KanaKanjiEntry(reading: "じぇー", surface: "J", frequency: 50),
        KanaKanjiEntry(reading: "けー", surface: "K", frequency: 50),
        KanaKanjiEntry(reading: "える", surface: "L", frequency: 50),
        KanaKanjiEntry(reading: "えむ", surface: "M", frequency: 50),
        KanaKanjiEntry(reading: "えぬ", surface: "N", frequency: 50),
        KanaKanjiEntry(reading: "おー", surface: "O", frequency: 50),
        KanaKanjiEntry(reading: "ぴー", surface: "P", frequency: 50),
        KanaKanjiEntry(reading: "きゅー", surface: "Q", frequency: 50),
        KanaKanjiEntry(reading: "あーる", surface: "R", frequency: 50),
        KanaKanjiEntry(reading: "えす", surface: "S", frequency: 50),
        KanaKanjiEntry(reading: "てぃー", surface: "T", frequency: 50),
        KanaKanjiEntry(reading: "ゆー", surface: "U", frequency: 50),
        KanaKanjiEntry(reading: "ぶい", surface: "V", frequency: 50),
        KanaKanjiEntry(reading: "だぶりゅー", surface: "W", frequency: 50),
        KanaKanjiEntry(reading: "えっくす", surface: "X", frequency: 50),
        KanaKanjiEntry(reading: "わい", surface: "Y", frequency: 50),
        KanaKanjiEntry(reading: "ぜっと", surface: "Z", frequency: 50)
    ]

    // Letter -> kana ("A" -> "えー"), derived from `entries`. Used to spell out acronyms in training text
    // ("GPU" -> "じーぴーゆー").
    private static let letterReadingTable: [String: String] = {
        var table: [String: String] = [:]
        for entry in entries {
            let surface = entry.surface
            if surface.count == 1, let scalar = surface.unicodeScalars.first {
                let v = scalar.value
                switch v {
                case 0x41...0x5A, 0x61...0x7A:
                    let upperKey = String(scalar).uppercased()
                    if table[upperKey] == nil {
                        table[upperKey] = entry.reading
                    }
                default:
                    break
                }
            }
        }
        return table
    }()

    // Latin-script surface -> most frequent reading ("wifi" -> "わいふぁい"), derived from `entries` and keyed by
    // width-folded lowercase. Used when building training labels so English words get their usual reading
    // instead of a romaji reading ("Apple" -> "あっぷる", not "あっぷれ").
    private static let alphabetReadingTable: [String: String] = {
        var table: [String: String] = [:]
        var maxFreq: [String: Int] = [:]
        for entry in entries {
            let key = entry.surface.folding(options: .widthInsensitive, locale: nil).lowercased()
            var isAlpha = false
            for s in key.unicodeScalars {
                if 0x61 <= s.value && s.value <= 0x7A {
                    isAlpha = true
                    break
                }
            }
            if isAlpha {
                let currentMax = maxFreq[key] ?? -1
                if currentMax < entry.frequency {
                    maxFreq[key] = entry.frequency
                    table[key] = entry.reading
                }
            }
        }
        return table
    }()

    /// Kana reading of a single Latin letter (full- or half-width).
    public static func readingForLetter(_ letter: String) -> String? {
        let key = letter.folding(options: .widthInsensitive, locale: nil).uppercased()
        return letterReadingTable[key]
    }

    /// Kana reading of a Latin-script word or abbreviation, ignoring width and case.
    public static func readingForAlphabetSurface(_ surface: String) -> String? {
        let key = surface.folding(options: .widthInsensitive, locale: nil).lowercased()
        return alphabetReadingTable[key]
    }
}
