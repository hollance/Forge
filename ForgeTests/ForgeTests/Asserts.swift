import Foundation

var verbose = false

func assertEqual<T>(_ a: T, _ b: T) where T:Equatable {
  if a != b {
    fatalError("Assertion failed: \(a) not equal to \(b)")
  }
}

func assertEqual<T>(_ a: [T], _ b: [T]) where T:Equatable {
  if a.count != b.count {
    fatalError("Assertion failed: array sizes not the same")
  }
  for i in 0..<a.count {
    if a[i] != b[i] {
      fatalError("Assertion failed: \(a[i]) not equal to \(b[i])")
    }
  }
}

func assertEqual(_ a: [Float], _ b: [Float], tolerance: Float) {
  if a.count != b.count {
    fatalError("Assertion failed: array sizes not the same")
  }
  var largestDiff: Float = 0
  var totalDiff: Float = 0
  for i in 0..<a.count {
    let diff = abs(a[i] - b[i])
    if diff > tolerance {
      fatalError("Assertion failed: difference too large at index \(i): \(a[i]) vs \(b[i])")
    }
    largestDiff = max(largestDiff, diff)
    totalDiff += diff
  }
  if verbose {
    print("    largest difference: \(largestDiff), average: \(totalDiff/Float(a.count))")
  }
}
