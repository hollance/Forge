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

import Metal
import MetalPerformanceShaders

/**
  Subtracts the mean red, green, blue colors from each input pixel.

  For each pixel this first multiplies all channels by `scale`, then subtracts
  `red`, `green`, and `blue` from the RGB channels; alpha is set to 0.
*/
public class SubtractMeanColor {
  let device: MTLDevice
  let pipeline: MTLComputePipelineState
  let params: [Float16]

  /**
    Creates a new SubtractMeanColor kernel for the specified colors.

    - Parameters:
      - scale: Used to scale up the pixels, which are typically in the range
        [0, 1] at this point. After scaling and subtracting the mean color,
        the pixels in the destination image will be in the range [-128, 128].
  */
  public init(device: MTLDevice, red: Float = 123.68, green: Float = 116.779,
              blue: Float = 103.939, scale: Float = 255) {
    self.device = device

    var float32 = [red, green, blue, 0, scale, scale, scale, 1]
    params = float32to16(&float32, count: float32.count)

    pipeline = makeFunction(device: device, name: "subtractMeanColor", useForgeLibrary: true)
  }

  public func encode(commandBuffer: MTLCommandBuffer,
                     sourceImage: MPSImage, destinationImage: MPSImage) {

    if let encoder = commandBuffer.makeComputeCommandEncoder() {
      encoder.setComputePipelineState(pipeline)
      encoder.setTexture(sourceImage.texture, index: 0)
      encoder.setTexture(destinationImage.texture, index: 1)
      encoder.setBytes(params, length: params.count * MemoryLayout<Float16>.stride, index: 0)
      encoder.dispatch(pipeline: pipeline, image: destinationImage)
      encoder.endEncoding()
    }

    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
  }
}

extension SubtractMeanColor: CustomKernel { }
