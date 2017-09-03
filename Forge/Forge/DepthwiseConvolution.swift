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

  - Note: On iOS 11 and up, use MPSCNNDepthWiseConvolutionDescriptor instead.
*/
public class DepthwiseConvolutionKernel: ForgeKernel {
  let pipeline: MTLComputePipelineState
  let weightsBuffer: MTLBuffer
  let biasBuffer: MTLBuffer

  /**
    Creates a new DepthwiseConvolution object.

    - Parameters:
      - channelMultiplier: If this is M, then each input channel has M kernels 
        applied to it, resulting in M output channels for each input channel.
        Default is 1.
      - relu: If true, applies a ReLU to the output. Default is false.
      - kernelWeights: The weights should be arranged in memory like this:
        `[featureChannels][kernelHeight][kernelWidth]`.
      - biasTerms: One bias term per channel (optional).
  */
  public init(device: MTLDevice,
              kernelWidth: Int,
              kernelHeight: Int,
              featureChannels: Int,
              strideInPixelsX: Int = 1,
              strideInPixelsY: Int = 1,
              channelMultiplier: Int = 1,
              neuronFilter: MPSCNNNeuron?,
              kernelWeights: UnsafePointer<Float>,
              biasTerms: UnsafePointer<Float>?) {

    precondition(kernelWidth == 3 && kernelHeight == 3, "Only 3x3 kernels are currently supported")
    precondition(channelMultiplier == 1, "Channel multipliers are not supported yet")

    let outputSlices = (featureChannels + 3) / 4
    let paddedOutputChannels = outputSlices * 4
    let count = paddedOutputChannels * kernelHeight * kernelWidth
    weightsBuffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)!

    let ptr = UnsafeMutablePointer(mutating: kernelWeights)
    let copyCount = featureChannels * kernelHeight * kernelWidth
    float32to16(input: ptr, output: weightsBuffer.contents(), count: copyCount)

    biasBuffer = makeBuffer(device: device,
                            channelFormat: .float16,
                            outputFeatureChannels: featureChannels,
                            biasTerms: biasTerms)

    var params = KernelParams()
    let constants = MTLFunctionConstantValues()
    configureNeuronType(filter: neuronFilter, constants: constants, params: &params)

    var stride = [ UInt16(strideInPixelsX), UInt16(strideInPixelsY) ]
    constants.setConstantValue(&stride, type: .ushort2, withName: "stride")

    let functionName: String
    if featureChannels <= 4 {
      functionName = "depthwiseConv3x3"
    } else {
      functionName = "depthwiseConv3x3_array"
    }
    pipeline = makeFunction(device: device, name: functionName,
                            constantValues: constants, useForgeLibrary: true)

    super.init(device: device, neuron: neuronFilter, params: params)
  }

  public override func encode(commandBuffer: MTLCommandBuffer,
                              sourceImage: MPSImage, destinationImage: MPSImage) {
    // TODO: set the KernelParams based on clipRect, destinationFeatureChannelOffset, edgeMode
    params.inputOffsetX = Int16(offset.x);
    params.inputOffsetY = Int16(offset.y);
    params.inputOffsetZ = Int16(offset.z);

    if let encoder = commandBuffer.makeComputeCommandEncoder() {
      encoder.setComputePipelineState(pipeline)
      encoder.setTexture(sourceImage.texture, index: 0)
      encoder.setTexture(destinationImage.texture, index: 1)
      encoder.setBytes(&params, length: MemoryLayout<KernelParams>.size, index: 0)
      encoder.setBuffer(weightsBuffer, offset: 0, index: 1)
      encoder.setBuffer(biasBuffer, offset: 0, index: 2)
      encoder.dispatch(pipeline: pipeline, image: destinationImage)
      encoder.endEncoding()
    }

    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
  }
}
