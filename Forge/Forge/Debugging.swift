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

  Writing `printChannelsForPixel(x: 5, y: 10, ...)` is the same as doing
  `print(layer_output[0, 10, 5, :])` in Python with layer output from Keras.
  Note that x and y are swapped in the Python code!

  To make sure the layer computes the right thing, feed the exact same image
  through Metal and Keras and compare the layer outputs.
*/
public func printChannelsForPixel(x: Int, y: Int, image: MPSImage) {
  let layerOutput = image.toFloatArray()
  print("Total size: \(layerOutput.count) floats")
  let w = image.width
  let h = image.height
  let s = (image.featureChannels + 3)/4
  for b in 0..<image.numberOfImages {
    for i in 0..<s {
      print("[batch index \(b), slice \(i) of \(s)]")
      for j in 0..<4 {
        print(layerOutput[b*s*h*w*4 + i*h*w*4 + y*w*4 + x*4 + j])
      }
    }
  }
}

/**
  Diagnostic tool for verifying that the neural network works correctly:
  prints out the pixel values for a given channel.
*/
@available(iOS 11.0, *)
public func printChannel(_ c: Int, image: MPSImage) {
  let count = image.featureChannels * image.height * image.width
  var dataFloat16 = [Float16](repeating: 0, count: count)
  image.readBytes(&dataFloat16, dataLayout: .featureChannelsxHeightxWidth, imageIndex: 0)

  let dataFloat32 = float16to32(&dataFloat16, count: count)

  let channelStride = image.height * image.width
  let heightStride = image.width
  let widthStride = 1

  for h in 0..<image.height {
    print("line \(h): ")
    for w in 0..<image.width {
      let value = dataFloat32[c*channelStride + h*heightStride + w*widthStride]
      print(value, terminator: ", ")
    }
    print("\n")
  }
}

/**
  Useful for checking that a computation gives the right answer, within the
  precision of 16-bit floats (which is only ~3 decimals).
  
  If the largest error is something like 0.000x and the average error is around
  1e-05 then you're good.
*/
public func verifySimilarResults(_ a: [Float], _ b: [Float], printSuspicious: Bool = true) {
  let count = min(a.count, b.count)
  if a.count != b.count {
    print("Array sizes are not the same: \(a.count) vs. \(b.count)")
  }

  var countSuspicious = 0
  var countNonZeroError = 0
  var largestError: Float = 0
  var largestErrorIndex = -1
  var averageError: Float = 0

  for i in 0..<count {
    let error = abs(a[i] - b[i])
    if error > largestError {
      largestError = error
      largestErrorIndex = i
    }
    if error != 0 {
      countNonZeroError += 1
    }
    if error > 0.01 {
      countSuspicious += 1
      if printSuspicious && countSuspicious <= 5 {
        print("\t\(i): \(a[i]) \t \(b[i]) \t \(error)")
      }
    }
    averageError += error
  }
  averageError /= Float(count)

  if largestErrorIndex == -1 {
    print("Arrays are identical")
  } else {
    print("Largest error: \(largestError) at index \(largestErrorIndex)")
    print("Average error: \(averageError)")
    print("Total suspicious entries: \(countSuspicious) out of \(count)")
    print("Total non-zero errors: \(countNonZeroError)")
  }
}
