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
import Metal
import MetalKit
import MetalPerformanceShaders

/* 
  Helper functions for creating the layers.

  The name of the layer is used to load its weights and bias values. You need
  to assign loader functions to weightsLoader and biasLoader.
*/

public var weightsLoader: ((String, Int) -> ParameterData?)?
public var biasLoader: ((String, Int) -> ParameterData?)?

/**
  Creates a convolution layer.
  
  - Parameters:
    - kernel: `(width, height)`
    - stride: `(x, y)`
*/
public func convolution(device: MTLDevice,
                        kernel: (Int, Int),
                        inChannels: Int,
                        outChannels: Int,
                        filter: MPSCNNNeuron,
                        name: String,
                        stride: (Int, Int) = (1, 1),
                        mergeOffset: Int = 0) -> MPSCNNConvolution {

  let countWeights = inChannels * kernel.1 * kernel.0 * outChannels
  let countBias = outChannels

  guard let weightsData = weightsLoader?(name, countWeights),
        let biasData = biasLoader?(name, countBias) else {
    fatalError("Error loading network parameters '\(name)'")
  }

  let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0,
                                         kernelHeight: kernel.1,
                                         inputFeatureChannels: inChannels,
                                         outputFeatureChannels: outChannels,
                                         neuronFilter: filter)
  desc.strideInPixelsX = stride.0
  desc.strideInPixelsY = stride.1

  let layer = MPSCNNConvolution(device: device,
                                convolutionDescriptor: desc,
                                kernelWeights: weightsData.pointer,
                                biasTerms: biasData.pointer,
                                flags: .none)
  layer.edgeMode = .zero
  layer.destinationFeatureChannelOffset = mergeOffset
  return layer
}

/**
  Creates a max-pooling layer.
*/
public func maxPooling(device: MTLDevice,
                       kernel: (Int, Int),
                       stride: (Int, Int),
                       mergeOffset: Int = 0) -> MPSCNNPoolingMax {
  let layer = MPSCNNPoolingMax(device: device,
                               kernelWidth: kernel.0,
                               kernelHeight: kernel.1,
                               strideInPixelsX: stride.0,
                               strideInPixelsY: stride.1)
  layer.offset = MPSOffset(x: kernel.0/2, y: kernel.1/2, z: 0)
  layer.edgeMode = .clamp
  layer.destinationFeatureChannelOffset = mergeOffset
  return layer
}

/**
  Creates an average-pooling layer.
*/
public func averagePooling(device: MTLDevice,
                           kernel: (Int, Int),
                           stride: (Int, Int),
                           mergeOffset: Int = 0) -> MPSCNNPoolingAverage {
  let layer = MPSCNNPoolingAverage(device: device,
                                   kernelWidth: kernel.0,
                                   kernelHeight: kernel.1,
                                   strideInPixelsX: stride.0,
                                   strideInPixelsY: stride.1)
  layer.offset = MPSOffset(x: kernel.0/2, y: kernel.1/2, z: 0)
  layer.edgeMode = .clamp
  layer.destinationFeatureChannelOffset = mergeOffset
  return layer
}

/**
  Creates a fully-connected layer that is connected to a convolution or 
  pooling layer.
  
  - Parameters:
    - shape: `(width, height)`. The spatial dimensions of the output image
      from the previous layer.
    - inChannels: The depth of the output image from the previous layer.
    - fanOut: The number of neurons in this layer.
*/
public func dense(device: MTLDevice,
                  shape: (Int, Int),
                  inChannels: Int,
                  fanOut: Int,
                  filter: MPSCNNNeuron?,
                  name: String,
                  mergeOffset: Int = 0) -> MPSCNNFullyConnected {

  let countWeights = inChannels * shape.0 * shape.1 * fanOut
  let countBias = fanOut

  guard let weightsData = weightsLoader?(name, countWeights),
        let biasData = biasLoader?(name, countBias) else {
    fatalError("Error loading network parameters '\(name)'")
  }

  // A fully-connected layer is a special version of a convolutional layer
  // where the kernel size is equal to the width/height of the input volume.
  // The output volume is 1x1xfanOut.
  let desc = MPSCNNConvolutionDescriptor(kernelWidth: shape.0,
                                         kernelHeight: shape.1,
                                         inputFeatureChannels: inChannels,
                                         outputFeatureChannels: fanOut,
                                         neuronFilter: filter)

  let layer = MPSCNNFullyConnected(device: device,
                                   convolutionDescriptor: desc,
                                   kernelWeights: weightsData.pointer,
                                   biasTerms: biasData.pointer,
                                   flags: .none)

  layer.destinationFeatureChannelOffset = mergeOffset
  return layer
}

/**
  Creates a fully-connected layer. Use this function when the new layer is 
  preceded by another fully-connected layer.
  
  - Parameters:
    - fanIn: The number of neurons in the previous fully-connected layer.
    - fanOut: The number of neurons in this layer.
*/
public func dense(device: MTLDevice,
                  fanIn: Int,
                  fanOut: Int,
                  filter: MPSCNNNeuron?,
                  name: String) -> MPSCNNFullyConnected {

  return dense(device: device, shape: (1, 1), inChannels: fanIn, fanOut: fanOut, filter: filter, name: name)
}

public enum PaddingType {
  case same    // add zero padding
  case valid   // don't add padding
}

extension MPSCNNConvolution {
  /**
    Computes the padding for a convolutional layer. You need to call this just
    before `convLayer.encode(...)` because it changes the layer's `offset`
    property.
    
    - Note: You really only need to call this when you don't want zero padding,
      or when the stride is not 1. For padding with stride 1, the default value
      of `(0, 0, 0)` for `self.offset` is sufficient. 
  */
  @nonobjc public func applyPadding(type: PaddingType, sourceImage: MPSImage, destinationImage: MPSImage) {
    if type == .same {
      let padH = (destinationImage.height - 1) * self.strideInPixelsY + self.kernelHeight - sourceImage.height
      let padW = (destinationImage.width  - 1) * self.strideInPixelsX + self.kernelWidth  - sourceImage.width
      self.offset = MPSOffset(x: (self.kernelWidth - padW)/2, y: (self.kernelHeight - padH)/2, z: 0)
    } else {
      self.offset = MPSOffset(x: self.kernelWidth/2, y: self.kernelHeight/2, z: 0)
    }
  }
}

/**
  Creates a depth-wise convolution layer.
  
  - Parameters:
    - kernel: `(width, height)`
    - stride: `(x, y)`
*/
public func depthwiseConvolution(device: MTLDevice,
                 kernel: (Int, Int),
                 channels: Int,
                 name: String,
                 stride: (Int, Int) = (1, 1)) -> DepthwiseConvolutionKernel {

  let countWeights = channels * kernel.1 * kernel.0

  guard let weightsData = weightsLoader?(name, countWeights) else {
    fatalError("Error loading network parameters '\(name)'")
  }

  return DepthwiseConvolutionKernel(device: device,
                                    kernelWidth: kernel.0,
                                    kernelHeight: kernel.1,
                                    featureChannels: channels,
                                    strideInPixelsX: stride.0,
                                    strideInPixelsY: stride.1,
                                    kernelWeights: weightsData.pointer)
}

/**
  Creates a 1x1 convolution layer.
*/
public func pointwiseConvolution(device: MTLDevice,
                                 inChannels: Int,
                                 outChannels: Int,
                                 filter: MPSCNNNeuron,
                                 name: String,
                                 stride: (Int, Int) = (1, 1),
                                 mergeOffset: Int = 0) -> MPSCNNConvolution {

  return convolution(device: device,
                     kernel: (1, 1),
                     inChannels: inChannels,
                     outChannels: outChannels,
                     filter: filter,
                     name: name)
}
