/*
  Copyright (c) 2017 M.I. Hollemans

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
import QuartzCore

public class FPSCounter {
  private(set) public var fps: Double = 0

  var frames = 0
  var startTime: CFTimeInterval = 0

  public func start() {
    frames = 0
    startTime = CACurrentMediaTime()
  }

  public func frameCompleted() {
    frames += 1
    let now = CACurrentMediaTime()
    let elapsed = now - startTime
    if elapsed > 0.1 {
      let current = Double(frames) / elapsed
      let smoothing = 0.75
      fps = smoothing*fps + (1 - smoothing)*current
      if elapsed > 1 {
        frames = 0
        startTime = CACurrentMediaTime()
      }
    }
  }
}
