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
  These values get passed to the compute kernel.
*/
public struct KernelParams {
  // The dimensions of the input image.
  var inputWidth: UInt16 = 0
  var inputHeight: UInt16 = 0
  var inputFeatureChannels: UInt16 = 0
  var inputSlices: UInt16 = 0

  // Where to start reading in the input image. From ForgeKernel's offset.
  var inputOffsetX: Int16 = 0
  var inputOffsetY: Int16 = 0
  var inputOffsetZ: Int16 = 0

  // The dimensions of the output image, derived from clipRect.size.
  var outputWidth: UInt16 = 0
  var outputHeight: UInt16 = 0
  var outputFeatureChannels: UInt16 = 0
  var outputSlices: UInt16 = 0

  // This is ForgeKernel's destinationFeatureChannelOffset divided by 4.
  var destinationSliceOffset: UInt16 = 0

  // Where to start writing in the output image, derived from clipRect.origin.
  var outputOffsetX: Int16 = 0
  var outputOffsetY: Int16 = 0
  var outputOffsetZ: Int16 = 0

  // Zero (0) or clamp (1).
  var edgeMode: UInt16 = 0

  // Additional parameters for MPSCNNNeurons.
  var neuronA: Float = 0
  var neuronB: Float = 0
}

/**
  Base class for compute kernels that need to work similarly to MPSCNNKernel.
*/
open class ForgeKernel: CustomKernel {
  public let device: MTLDevice
  public let neuron: MPSCNNNeuron?

  public var offset = MPSOffset(x: 0, y: 0, z: 0)
  public var clipRect = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),    // MPSRectNoClip
                                  size: MTLSize(width: -1, height: -1, depth: -1))
  public var destinationFeatureChannelOffset = 0
  public var edgeMode = MPSImageEdgeMode.zero

  var params = KernelParams()

  public init(device: MTLDevice, neuron: MPSCNNNeuron?, params: KernelParams) {
    self.device = device
    self.neuron = neuron
    self.params = params
  }

  public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    fatalError("Subclass must implement this function")
  }
}

func configureNeuronType(filter: MPSCNNNeuron?,
                                constants: MTLFunctionConstantValues,
                                params: inout KernelParams) {
  var neuronType: UInt16 = 0
  if let filter = filter as? MPSCNNNeuronReLU {
    neuronType = 1
    params.neuronA = filter.a
  } else if let filter = filter as? MPSCNNNeuronLinear {
    neuronType = 2
    params.neuronA = filter.a
    params.neuronB = filter.b
  } else if filter is MPSCNNNeuronSigmoid {
    neuronType = 3
  } else if let filter = filter as? MPSCNNNeuronTanH {
    neuronType = 4
    params.neuronA = filter.a
    params.neuronB = filter.b
  } else if filter is MPSCNNNeuronAbsolute {
    neuronType = 5
  }
  constants.setConstantValue(&neuronType, type: .ushort, withName: "neuronType")
}
