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

public enum Math {

  /**
    Returns a number between 0 and count, using the probabilities in x.
  */
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

  /**
    Computes the "softmax" function over an array.

    Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

    This is what softmax looks like in "pseudocode" (actually using Python
    and numpy):

        x -= np.max(x)
        exp_scores = np.exp(x)
        softmax = exp_scores / np.sum(exp_scores)

    First we shift the values of x so that the highest value in the array is 0. 
    This ensures numerical stability with the exponents, so they don't blow up.
  */
  public static func softmax(_ x: [Float]) -> [Float] {
    var x = x
    let len = vDSP_Length(x.count)

    // Find the maximum value in the input array.
    var max: Float = 0
    vDSP_maxv(x, 1, &max, len)

    // Subtract the maximum from all the elements in the array.
    // Now the highest value in the array is 0.
    max = -max
    vDSP_vsadd(x, 1, &max, &x, 1, len)

    // Exponentiate all the elements in the array.
    var count = Int32(x.count)
    vvexpf(&x, x, &count)

    // Compute the sum of all exponentiated values.
    var sum: Float = 0
    vDSP_sve(x, 1, &sum, len)

    // Divide each element by the sum. This normalizes the array contents
    // so that they all add up to 1.
    vDSP_vsdiv(x, 1, &sum, &x, 1, len)

    return x
  }

  /**
    Logistic sigmoid.
  */
  public static func sigmoid(_ x: Float) -> Float {
    return 1 / (1 + exp(-x))
  }
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(a: CGRect, b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}
