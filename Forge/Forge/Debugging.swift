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
import MetalPerformanceShaders

/**
  Diagnostic tool for verifying that the neural network works correctly:
  prints out the channels for a given pixel coordinate.

  Writing `printChannelsForPixel(x: 10, y: 10, ...)` is the same as doing
  `print(layer_output[0, 10, 10, :])` in Python with layer output from Keras.

  To make sure the layer computes the right thing, feed the exact same image
  through Metal and Keras and compare the layer outputs.
*/
public func printChannelsForPixel(x: Int, y: Int, image: MPSImage) {
  let layerOutput = image.toFloatArray()
  print("Total size: \(layerOutput.count) floats")
  let w = image.width
  let h = image.height
  let c = image.featureChannels
  for i in 0..<(c + 3)/4 {
    for j in 0..<4 {
      print(layerOutput[i*h*w*4 + y*w*4 + x*4 + j])
    }
  }
}
