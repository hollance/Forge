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
import MetalPerformanceShaders

/**
  The abstract base class for all layers. You cannot create instances of this 
  class directly.
  
  When you create a model, you're actually building a graph of these Layer
  objects.
*/
open class Layer: CustomDebugStringConvertible {
  var name: String
  var next: Layer?

  // Most layers require that the complete shape of the input texture is known.
  // However, some layers (such as Resize and Custom) can handle inputs of any
  // size. If your first layer is a type that must know the size (such as Conv)
  // then you need to specify that size with an Input layer.
  var allowsIncompleteShape = false

  /**
    Whether this layer writes its results into an MPSTemporaryImage. Normally
    this is true for all layers except the last. You can override this if you
    want to keep track of the layer's MPSImage for processing afterwards.
  */
  public var temporaryImage = true

  fileprivate init(name: String = "") {
    self.name = name
  }

  public var debugDescription: String {
    return name
  }

  /* Subclasses must implement these methods. */

  var typeName: String {
    fatalError("Subclass must implement this function")
  }

  func outputShape(for inputShape: DataShape) -> DataShape {
    fatalError("Subclass must implement this function")
  }

  func createCompute(device: MTLDevice, weights: ParameterData?, biases: ParameterData?) -> Compute? {
    fatalError("Subclass must implement this function")
  }

  // If these return 0, we won't attempt to load any parameters for the layer.
  var weightCount: Int { return 0 }
  var biasCount: Int { return 0 }

  /* The things below this line are filled in by compile(). */

  var inputShape = DataShape()
  var outputShape = DataShape()
  var compute: Compute?
}

/**
  A placeholder for input. This layer doesn't do anything, but can be used to
  force the input texture to be in a specific shape.
  
  If your first layer is `Resize`, which takes a texture of arbitrary size and
  scales it to a fixed size, then you don't really need `Input`.
  
  However, if your first layer is something like a `Convolution`, then you need
  `Input` to specify the size of the texture that goes into the conv layer. 
  (Without it, we won't know how large the `Convolution` layer's output will be
  and as a result we can't allocate an MPSTemporaryImage for it.)
*/
public class Input: Layer {
  public init(width: Int? = nil, height: Int? = nil, channels: Int? = nil, name: String = "") {
    super.init(name: name)
    self.inputShape = DataShape(width: width ?? -1,
                                height: height ?? -1,
                                channels: channels ?? -1)

    if !self.inputShape.isFullySpecified {
      allowsIncompleteShape = true
    }
  }

  override var typeName: String {
    return "Input"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }
}

/**
  Convolutional layer.
*/
public class Convolution: Layer {
  let kernel: (Int, Int)
  let channels: Int
  let filter: MPSCNNNeuron?
  let padding: Bool
  let stride: (Int, Int)

  /**
    Creates a convolution layer.
  
    - Parameters:
      - kernel: `(width, height)`
      - channels: number of output channels
      - stride: `(x, y)`
      - name: used to load the layer's parameters
  */
  public init(kernel: (Int, Int),
              channels: Int,
              filter: MPSCNNNeuron? = nil,
              padding: Bool = true,
              stride: (Int, Int) = (1, 1),
              name: String = "") {
    self.kernel = kernel
    self.channels = channels
    self.filter = filter
    self.padding = padding
    self.stride = stride
    super.init(name: name)
  }

  override var typeName: String {
    return "Conv"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    if padding {
      return DataShape(width: inputShape.width,
                      height: inputShape.height,
                    channels: channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  inputShape.channels)
    }
  }

  override var weightCount: Int {
    return inputShape.channels * kernel.1 * kernel.0 * outputShape.channels
  }

  override var biasCount: Int {
    return outputShape.channels
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {

    guard let weights = weights, let biases = biases else {
      fatalError("Compile error: missing parameters for layer '\(name)'")
    }

    let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: outputShape.channels,
                                           neuronFilter: filter)
    desc.strideInPixelsX = stride.0
    desc.strideInPixelsY = stride.1

    let layer = MPSCNNConvolution(device: device,
                                  convolutionDescriptor: desc,
                                  kernelWeights: weights.pointer,
                                  biasTerms: biases.pointer,
                                  flags: .none)
    layer.edgeMode = .zero

    if padding {
      let padH = (outputShape.height - 1) * layer.strideInPixelsY + layer.kernelHeight - inputShape.height
      let padW = (outputShape.width  - 1) * layer.strideInPixelsX + layer.kernelWidth  - inputShape.width
      layer.offset = MPSOffset(x: (layer.kernelWidth - padW)/2, y: (layer.kernelHeight - padH)/2, z: 0)
    } else {
      layer.offset = MPSOffset(x: layer.kernelWidth/2, y: layer.kernelHeight/2, z: 0)
    }

    return Compute.mpscnn(layer)
  }
}

/**
  Abstract base class for max-pooling and average-pooling layers.
*/
public class Pooling: Layer {
  let kernel: (Int, Int)
  let stride: (Int, Int)

  public init(kernel: (Int, Int), stride: (Int, Int), name: String = "") {
    self.kernel = kernel
    self.stride = stride
    super.init(name: "")
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                    height: (inputShape.height - kernel.1) / stride.1 + 1,
                  channels:  inputShape.channels)
  }
}

/**
  Max-pooling layer.
*/
public class MaxPooling: Pooling {
  override var typeName: String {
    return "MaxPool"
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {
    // FUTURE: Since pooling layers don't have parameters, it makes sense to
    // reuse them where possible. Put them into a dictionary using the kernel
    // size and stride as key.
    let layer = MPSCNNPoolingMax(device: device,
                                 kernelWidth: kernel.0,
                                 kernelHeight: kernel.1,
                                 strideInPixelsX: stride.0,
                                 strideInPixelsY: stride.1)
    layer.offset = MPSOffset(x: kernel.0/2, y: kernel.1/2, z: 0)
    layer.edgeMode = .clamp
    return Compute.mpscnn(layer)
  }
}

/**
  Average-pooling layer.
*/
public class AveragePooling: Pooling {
  override var typeName: String {
    return "AvgPool"
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {
    // FUTURE: Also recycle these. See note in max-pool layer.
    let layer = MPSCNNPoolingAverage(device: device,
                                     kernelWidth: kernel.0,
                                     kernelHeight: kernel.1,
                                     strideInPixelsX: stride.0,
                                     strideInPixelsY: stride.1)
    layer.offset = MPSOffset(x: kernel.0/2, y: kernel.1/2, z: 0)
    layer.edgeMode = .clamp
    return Compute.mpscnn(layer)
  }
}

/**
  Fully-connected layer.
*/
public class Dense: Layer {
  let neurons: Int
  let filter: MPSCNNNeuron?

  /**
    Creates a fully-connected layer.
  
    - Parameters:
      - neurons: the number of neurons in this layer
      - name: used to load the layer's parameters
  */
  public init(neurons: Int, filter: MPSCNNNeuron? = nil, name: String = "") {
    self.neurons = neurons
    self.filter = filter
    super.init(name: name)
  }

  override var typeName: String {
    return "Dense"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: 1, height: 1, channels: neurons)
  }

  override var weightCount: Int {
    return inputShape.channels * inputShape.height * inputShape.width * neurons
  }

  override var biasCount: Int {
    return neurons
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {

    guard let weights = weights, let biases = biases else {
      fatalError("Compile error: missing parameters for layer '\(name)'")
    }

    // A fully-connected layer is a special version of a convolutional layer
    // where the kernel size is equal to the width/height of the input volume.
    // The output volume is 1x1xfanOut.
    let desc = MPSCNNConvolutionDescriptor(kernelWidth: inputShape.width,
                                           kernelHeight: inputShape.height,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: neurons,
                                           neuronFilter: filter)

    let layer = MPSCNNFullyConnected(device: device,
                                     convolutionDescriptor: desc,
                                     kernelWeights: weights.pointer,
                                     biasTerms: biases.pointer,
                                     flags: .none)
    return Compute.mpscnn(layer)
  }
}

/**
  Softmax layer.
*/
public class Softmax: Layer {
  public override init(name: String = "") {
    super.init(name: name)
  }

  override var typeName: String {
    return "Softmax"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {
    return Compute.mpscnn(MPSCNNSoftMax(device: device))
  }
}

/**
  Lets you use any MPSCNNNeuron as a layer of its own.
*/
public class Activation: Layer {
  let filter: MPSCNNNeuron

  public init(_ filter: MPSCNNNeuron, name: String = "") {
    self.filter = filter
    super.init(name: name)
  }

  override var typeName: String {
    return "Activation"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {
    return Compute.mpscnn(filter)
  }
}

/**
  Resizes the input texture to a specific size. The input is expected to have
  3 or 4 channels. Always outputs a 3-channel image.
*/
public class Resize: Layer {
  let width: Int
  let height: Int

  public init(width: Int, height: Int, name: String = "") {
    self.width = width
    self.height = height
    super.init(name: name)
    self.allowsIncompleteShape = true
  }

  override var typeName: String {
    return "Resize"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width, height: height, channels: 3)
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {
    return Compute.mps(MPSImageLanczosScale(device: device))
  }
}

/**
  The Custom layer type accepts any object that conforms to this protocol.

  - NOTE: The `encode()` function must do the following:
  
          // Let Metal know the temporary image can be recycled.
          if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
          }
*/
public protocol CustomKernel {
  func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage)
}

/**
  Use this to call your own compute kernels.
*/
public class Custom: Layer {
  let custom: CustomKernel
  let width: Int?
  let height: Int?
  let channels: Int?

  /**
    Creates a new layer using a custom compute kernel.

    - Note: If `width`, `height`, or `channels` is nil, then that dimension
      from the input shape is passed through unchanged.
  */
  public init(_ custom: CustomKernel,
              width: Int? = nil,
              height: Int? = nil,
              channels: Int? = nil,
              name: String = "") {
    self.custom = custom
    self.width = width
    self.height = height
    self.channels = channels
    super.init(name: name)

    // If the output shape is completely specified, then this layer accepts
    // any input, even if some dimensions are unknown.
    if width != nil && height != nil && channels != nil {
      allowsIncompleteShape = true
    }
  }

  override var typeName: String {
    return "Custom"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width ?? inputShape.width,
                     height: height ?? inputShape.height,
                     channels: channels ?? inputShape.channels)
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) -> Compute? {
    return Compute.custom(custom)
  }
}
