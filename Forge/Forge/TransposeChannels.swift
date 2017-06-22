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
import Accelerate

/**
  Transposes the channels inside an MPSImage.
*/
public class TransposeChannelsKernel {
  let device: MTLDevice
  let pipeline: MTLComputePipelineState
  let buffer: MTLBuffer

  /**
    Creates a new TransposeChannelKernel object.

    - Parameters:
      - featureChannels: The number of channels in the input image. The output
        image will have the same number of channels.
      - permute: A list of channels to permute. (The same channel index is
        allowed to appear more than once.)
  */
  public init(device: MTLDevice,
              featureChannels: Int,
              permute: [UInt16]) {

    self.device = device

    let slices = (featureChannels + 3) / 4
    buffer = device.makeBuffer(length: MemoryLayout<UInt16>.stride * slices * 4)!
    memcpy(buffer.contents(), permute, MemoryLayout<UInt16>.stride * featureChannels)

    // If there's more than one texture slice in the image, we have to use a
    // kernel that uses texture2d_array objects.
    let functionName: String
    if featureChannels <= 4 {
      functionName = "transposeChannels"
    } else {
      functionName = "transposeChannels_array"
    }
    pipeline = makeFunction(device: device, name: functionName, useForgeLibrary: true)
  }

  public func encode(commandBuffer: MTLCommandBuffer,
                     sourceImage: MPSImage, destinationImage: MPSImage) {
    if let encoder = commandBuffer.makeComputeCommandEncoder() {
      encoder.setComputePipelineState(pipeline)
      encoder.setTexture(sourceImage.texture, index: 0)
      encoder.setTexture(destinationImage.texture, index: 1)
      encoder.setBuffer(buffer, offset: 0, index: 0)
      encoder.dispatch(pipeline: pipeline, image: destinationImage)
      encoder.endEncoding()
    }

    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
  }
}
