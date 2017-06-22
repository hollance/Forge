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
  Simple wrapper around a compute shader.
*/
open class SimpleKernel {
  let device: MTLDevice
  let pipeline: MTLComputePipelineState
  let name: String

  public init(device: MTLDevice, functionName: String, useForgeLibrary: Bool = false) {
    self.device = device
    self.name = functionName
    pipeline = makeFunction(device: device, name: functionName, useForgeLibrary: useForgeLibrary)
  }

  public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    if let encoder = commandBuffer.makeComputeCommandEncoder() {
      encoder.pushDebugGroup(name)
      encoder.setComputePipelineState(pipeline)
      encoder.setTexture(sourceImage.texture, index: 0)
      encoder.setTexture(destinationImage.texture, index: 1)
      encoder.dispatch(pipeline: pipeline, image: destinationImage)
      encoder.popDebugGroup()
      encoder.endEncoding()
    }

    // Let Metal know the temporary image can be recycled.
    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
  }
}

extension SimpleKernel: CustomKernel { }
