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
}
