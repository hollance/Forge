/*
  Copyright (c) 2016-2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import Foundation
import Accelerate

enum Math {
  /* Returns a number between 0 and count, using the probabilities in x. */
  public static func randomlySample(_ x: UnsafeMutablePointer<Float>, _ count: Int) -> Int {
    // Compute the cumulative sum of the probabilities.
    var cumsum = [Float](repeating: 0, count: count)
    var sum: Float = 0
    for i in 0..<count {
      sum += x[i]
      cumsum[i] = sum
    }

    // Normalize so that the last element is exactly 1.0.
    var last = cumsum.last!
    vDSP_vsdiv(cumsum, 1, &last, &cumsum, 1, vDSP_Length(count))

    // Get a new random number between 0 and 1 (exclusive).
    let sample = Random.random()

    // Find the index of where sample would go in the array.
    for i in stride(from: count - 2, through: 0, by: -1) {
      if cumsum[i] <= sample {
        return i + 1
      }
    }
    return 0
  }
}
