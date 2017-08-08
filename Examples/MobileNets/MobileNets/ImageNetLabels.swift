import Foundation

/**
  The list of ImageNet label names, loaded from synset_words.txt.
*/
public class ImageNetLabels {
  private var labels = [String](repeating: "", count: 1000)

  public init() {
    if let path = Bundle.main.path(forResource: "synset_words", ofType: "txt") {
      for (i, line) in lines(filename: path).enumerated() {
        if i < 1000 {
          // Strip off the WordNet ID (the first 10 characters).
          labels[i] = String(line[line.index(line.startIndex, offsetBy: 10)...])
        }
      }
    }
  }

  private func lines(filename: String) -> [String] {
    do {
      let text = try String(contentsOfFile: filename, encoding: .ascii)
      let lines = text.components(separatedBy: NSCharacterSet.newlines)
      return lines
    } catch {
      fatalError("Could not load file: \(filename)")
    }
  }

  public subscript(i: Int) -> String {
    return labels[i]
  }
}
