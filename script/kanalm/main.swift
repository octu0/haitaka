import Foundation
import Haitaka

// Writes the manifest transcripts as kana, one sentence per line, as input for the kana n-gram LM (train --kana-lm).
// Usage: kanalm --english-dict <cmudict> [--exclude <eval manifest>]... <train manifest>... > kana_corpus.txt
var englishDictPath = ""
var excludePaths: [String] = []
var manifestPaths: [String] = []
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    switch args[argIdx] {
    case "--english-dict":
        englishDictPath = args[argIdx + 1]
        argIdx += 1
    case "--exclude":
        excludePaths.append(args[argIdx + 1])
        argIdx += 1
    default:
        manifestPaths.append(args[argIdx])
    }
    argIdx += 1
}
guard englishDictPath.isEmpty != true, let englishDict = try? EnglishPronunciations(contentsOfFile: englishDictPath) else {
    FileHandle.standardError.write("Specify an English pronunciation dictionary with --english-dict\n".data(using: .utf8)!)
    exit(1)
}
func paths(in manifest: String) -> [(path: String, text: String)] {
    guard let data = FileManager.default.contents(atPath: manifest), let whole = String(data: data, encoding: .utf8) else {
        return []
    }
    var rows: [(String, String)] = []
    for line in whole.split(separator: "\n") {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let path = obj["path"] as? String, let text = obj["text"] as? String else {
            continue
        }
        rows.append((path, text))
    }
    return rows
}
var excluded: Set<String> = []
for p in excludePaths {
    for row in paths(in: p) {
        excluded.insert(row.path)
    }
}
let converter = KanjiConverter(english: englishDict)
var written = 0
var skipped = 0
for p in manifestPaths {
    for row in paths(in: p) {
        if excluded.contains(row.path) {
            skipped += 1
            continue
        }
        let kana = converter.convertToHiragana(row.text)
        if kana.isEmpty {
            continue
        }
        print(kana)
        written += 1
    }
}
FileHandle.standardError.write("\(written) kana lines (\(skipped) excluded as evaluation data)\n".data(using: .utf8)!)
