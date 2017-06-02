import Foundation
import Forge

class ArrayTests {
  func testArgmax() {
    print("\(self).\(#function)")

    let a: [Float] = [ 2, 0, 7, -1, 8, 3, 7, 5 ]

    let (maxIndex, maxValue) = a.argmax()
    assertEqual(maxIndex, 4)
    assertEqual(maxValue, 8)
  }

  func testArgsort() {
    print("\(self).\(#function)")

    let a = [ 2, 0, 7, -1, 8, 3, -2, 5 ]
    let s = a.argsort(by: <)

    let i = [ 6, 3, 1, 0, 5, 7, 2, 4 ]   // these are indices!
    assertEqual(s, i)
  }

  func testGather() {
    print("\(self).\(#function)")

    let a = [ 2, 0, 7, -1, 8, 3, -2, 5 ]
    let i = [ 6, 3, 1, 0, 5, 7, 2, 4 ]   // these are indices!
    let g = a.gather(indices: i)

    let e = [ -2, -1, 0, 2, 3, 5, 7, 8 ]
    assertEqual(g, e)
  }
}
