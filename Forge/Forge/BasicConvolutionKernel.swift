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
  Just your regular kind of convolution. There is no reason to use this class
  in production code, use MPSCNNConvolution instead. This class exists only to
  experiment with Forge features.
*/
public class BasicConvolutionKernel: ForgeKernel {
  let pipeline: MTLComputePipelineState
  let weightsBuffer: MTLBuffer
  let biasBuffer: MTLBuffer

  public init(device: MTLDevice,
              kernelWidth: Int,
              kernelHeight: Int,
              inputFeatureChannels: Int,
              outputFeatureChannels: Int,
              strideInPixelsX: Int = 1,
              strideInPixelsY: Int = 1,
              neuronFilter: MPSCNNNeuron?,
              kernelWeights: UnsafePointer<Float>,
              biasTerms: UnsafePointer<Float>?) {

    precondition(kernelWidth == 3 && kernelHeight == 3, "Only 3x3 kernels are currently supported")

    // Convert the weights to 16-bit floats and copy them into a Metal buffer.
    weightsBuffer = makeBuffer(device: device,
                               channelFormat: .float16,
                               kernelWidth: kernelWidth,
                               kernelHeight: kernelHeight,
                               inputFeatureChannels: inputFeatureChannels,
                               outputFeatureChannels: outputFeatureChannels,
                               weights: kernelWeights)

    biasBuffer = makeBuffer(device: device,
                            channelFormat: .float16,
                            outputFeatureChannels: outputFeatureChannels,
                            biasTerms: biasTerms)

    // Specialize the compute function, so that the Metal compiler will build
    // a unique kernel based on the chosen options for stride, etc. We could
    // pass these options into the kernel using a buffer instead, but then we
    // would have to branch at runtime, which is slower.
    var params = KernelParams()
    let constants = MTLFunctionConstantValues()
    configureNeuronType(filter: neuronFilter, constants: constants, params: &params)

    var stride = [ UInt16(strideInPixelsX), UInt16(strideInPixelsY) ]
    constants.setConstantValue(&stride, type: .ushort2, withName: "stride")

    // If there's more than one texture slice in the image, we have to use a
    // kernel that uses texture2d_array objects.
    let functionName: String
    if outputFeatureChannels <= 4 {
      functionName = "conv3x3"
    } else {
      functionName = "conv3x3_array"
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
