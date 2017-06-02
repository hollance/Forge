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

public enum Random {
  /** Returns a new random number in the range [0, upperBound) (exclusive). */
  public static func random(_ upperBound: Int) -> Int {
    return Int(arc4random_uniform(UInt32(upperBound)))
  }

  /** Returns a new random number in the range [0, 1) (exclusive). */
  public static func random() -> Float {
    return Float(arc4random()) / Float(0x100000000)
  }

  /**
    Fills up the given array with uniformly random values between -scale
    and +scale.
  */
  public static func uniformRandom(_ x: UnsafeMutablePointer<Float>,
                                   count: Int, scale: Float) {
    for i in 0..<count {
      x[i] = (random()*2 - 1) * scale
    }
  }

  /**
    Fills up the given array with uniformly random values between -scale
    and +scale. You can seed the random generator to create reproducible
    results.
  */
  public static func uniformRandom(_ x: UnsafeMutablePointer<Float>,
                                   count: Int, scale: Float, seed: Int) {
    srand48(seed)
    for i in 0..<count {
      x[i] = Float(drand48()*2 - 1) * scale
    }
  }
}
