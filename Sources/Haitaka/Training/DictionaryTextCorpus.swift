import Foundation

/// Loads text without audio to extend the kana-kanji conversion dictionary.
///
/// Accepts UTF-8 `.txt` (one sentence per line), `.srt` (cue text only), and directories containing them (recursive).
public enum DictionaryTextCorpus {
    /// Collects non-empty lines from a file or directory. Unreadable files are skipped.
    public static func load(path: String) -> [String] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
            return []
        }
        if isDirectory.boolValue {
            return loadDirectory(path)
        }
        return loadFile(path)
    }

    static func loadDirectory(_ dir: String) -> [String] {
        guard let enumerator = FileManager.default.enumerator(atPath: dir) else {
            return []
        }
        var files: [String] = []
        while let relative = enumerator.nextObject() as? String {
            let lower = relative.lowercased()
            if lower.hasSuffix(".txt") || lower.hasSuffix(".srt") {
                files.append((dir as NSString).appendingPathComponent(relative))
            }
        }
        files.sort()
        var lines: [String] = []
        for file in files {
            lines.append(contentsOf: loadFile(file))
        }
        return lines
    }

    static func loadFile(_ path: String) -> [String] {
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else {
            return []
        }
        let content = normalizedNewlines(raw)
        if path.lowercased().hasSuffix(".srt") {
            return srtCueTexts(content)
        }
        var lines: [String] = []
        for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty != true {
                lines.append(trimmed)
            }
        }
        return lines
    }

    /// Strips the BOM and converts CRLF/CR to LF.
    static func normalizedNewlines(_ text: String) -> String {
        return text
            .replacingOccurrences(of: "\u{FEFF}", with: "")
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }

    /// Returns one line per SRT cue, dropping index and timing lines. Multi-line cue text is joined with spaces.
    static func srtCueTexts(_ content: String) -> [String] {
        var cues: [String] = []
        var body: [String] = []
        var inCue = false
        func closeCue() {
            if inCue && body.isEmpty != true {
                cues.append(body.joined(separator: " "))
            }
            inCue = false
            body = []
        }
        for rawLine in content.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                closeCue()
                continue
            }
            if line.contains("-->") {
                closeCue()
                inCue = true
                continue
            }
            if inCue {
                body.append(line)
            }
        }
        closeCue()
        return cues
    }
}
