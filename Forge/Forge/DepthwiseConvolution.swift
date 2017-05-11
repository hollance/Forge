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
  Depth-wise convolution
  
  Applies a different convolution kernel to each input channel. Only a single
  kernel is applied to each input channel and so the number of output channels
  is the same as the number of input channels.

  A depth-wise convolution only performs filtering; it doesn't combine channels
  to create new features like a regular convolution does.
*/
public class DepthwiseConvolutionKernel {
  let device: MTLDevice
  let pipeline: MTLComputePipelineState
  let weightsBuffer: MTLBuffer

  /**
    Creates a new DepthwiseConvolution object.

    - Parameters:
      - channelMultiplier: If this is M, then each input channel has M kernels 
        applied to it, resulting in M output channels for each input channel.
        Default is 1.
      - relu: If true, applies a ReLU to the output. Default is false.
      - kernelWeights: The weights should be arranged in memory like this:
        `[kernelHeight][kernelWidth][featureChannels]`. There is no bias for
        this kind of layer.
  */
  public init(device: MTLDevice,
              kernelWidth: Int,
              kernelHeight: Int,
              featureChannels: Int,
              strideInPixelsX: Int = 1,
              strideInPixelsY: Int = 1,
              channelMultiplier: Int = 1,
              relu: Bool = false,
              kernelWeights: UnsafePointer<Float>) {

    precondition(kernelWidth == 3 && kernelHeight == 3, "Only 3x3 kernels are currently supported")
    precondition(channelMultiplier == 1, "Channel multipliers are not supported yet")

    self.device = device

    // Convert the weights to 16-bit floats and copy them into a Metal buffer.
    let inputSlices = (featureChannels + 3) / 4
    let paddedInputChannels = inputSlices * 4
    let count = kernelHeight * kernelWidth * paddedInputChannels
    weightsBuffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)

    copy(weights: kernelWeights, to: weightsBuffer, channelFormat: .float16,
         kernelWidth: kernelWidth, kernelHeight: kernelHeight,
         inputFeatureChannels: featureChannels, outputFeatureChannels: 1)

    // Specialize the compute function, so that the Metal compiler will build
    // a unique kernel based on the chosen options for stride, etc. We could
    // pass these options into the kernel using a buffer instead, but then we
    // would have to branch at runtime, which is slower.
    var stride = [ UInt16(strideInPixelsX), UInt16(strideInPixelsY) ]
    var useReLU = relu
    let values = MTLFunctionConstantValues()
    values.setConstantValue(&stride, type: .ushort2, at: 2)
    values.setConstantValue(&useReLU, type: .bool, at: 3)

    // If there's more than one texture slice in the image, we have to use a
    // kernel that uses texture2d_array objects.
    let functionName: String
    if featureChannels <= 4 {
      functionName = "depthwiseConv3x3_half"
    } else {
      functionName = "depthwiseConv3x3_half_array"
    }
    pipeline = makeFunction(device: device, name: functionName, constantValues: values, useForgeLibrary: true)
  }

  public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    let encoder = commandBuffer.makeComputeCommandEncoder()
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(sourceImage.texture, at: 0)
    encoder.setTexture(destinationImage.texture, at: 1)
    encoder.setBuffer(weightsBuffer, offset: 0, at: 0)
    encoder.dispatch(pipeline: pipeline, image: destinationImage)
    encoder.endEncoding()

    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
  }
}
