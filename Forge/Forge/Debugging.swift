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

/**
  Creates an MPSImage from a file containing binary 32-floats.
  
  The data must be arranged as follows:
  
  - For each image in the batch, first there must be channels 0-3 for all 
    pixels, followed by channels 4-7, followed by channels 8-11, etc.
    This is so we can copy the data into the MTLTextures slice-by-slice.

  In Python, given an image of shape `(batchSize, height, width, channels)` 
  you can convert it as follows to the expected format:

      X = np.zeros((batchSize, height, width, channels)).astype(np.float32)
      slices = []
      for j in range(X.shape[0]):
          for i in range(X.shape[3] // 4):
              slice = X[j, :, :, i*4:(i+1)*4].reshape(1, X.shape[1], X.shape[2], 4)
              slices.append(slice)
      X_metal = np.concatenate(slices, axis=0)
      X_metal.tofile("X.bin")
*/
public func loadImage(device: MTLDevice,
                      url: URL,
                      batchSize: Int = 1,
                      width: Int,
                      height: Int,
                      channels: Int) -> MPSImage {
  let imageDesc = MPSImageDescriptor(channelFormat: .float16,
                                     width: width,
                                     height: height,
                                     featureChannels: channels,
                                     numberOfImages: batchSize,
                                     usage: [.shaderRead, .shaderWrite])
  let image = MPSImage(device: device, imageDescriptor: imageDesc)

  let data = try! Data(contentsOf: url)
  let inputSize = batchSize * height * width * channels
  var float32Input = [Float](repeating: 0, count: inputSize)
  _ = float32Input.withUnsafeMutableBufferPointer { ptr in
    data.copyBytes(to: ptr)
  }

  let float16Input = float32to16(&float32Input, count: inputSize)
  float16Input.withUnsafeBytes { buf in
    // We copy 4 channels worth of data at a time
    let bytesPerRow = width * 4 * MemoryLayout<Float16>.stride
    let slices = ((channels + 3) / 4) * batchSize
    var ptr = buf.baseAddress!
    for s in 0..<slices {
      image.texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            slice: s,
                            withBytes: ptr,
                            bytesPerRow: bytesPerRow,
                            bytesPerImage: 0)
      ptr += height * bytesPerRow
    }
  }

  return image
}
