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

  // Whether this layer actually computes anything.
  var needsEncoding = true

  // Most layers take MPSImages as input but for Resize it's more optimal to
  // work directly on the input texture. That saves making an MPSImage object.
  // Probably a premature optimization. ;-)
  var wantsTextures = false

  // Most layers require that the complete shape of the input texture is known.
  // However, some layers (such as Resize and Custom) can handle inputs of any
  // size. If your first layer is a type that must know the size (Convolution)
  // then you need to specify that size with an Input layer.
  var allowsIncompleteShape = false

  // Used to set destinationFeatureChannelOffset for merging the output from
  // multiple layers into one image. If needsDestinationImage is true, a new
  // (temporary) image is allocated for this layer; if false, the layer will
  // write into the destination image from its enclosing Group.
  var mergeOffset = 0
  var needsDestinationImage = true

  // These are filled in by the compiler.
  var inputShape = DataShape()
  var outputShape = DataShape()

  /**
    Whether this layer writes its results into an MPSTemporaryImage. Normally
    this is true for all layers except the last. You can override this if you
    want to keep track of the layer's MPSImage for processing afterwards.
  */
  public var imageIsTemporary = true

  fileprivate init(name: String = "") {
    self.name = name
  }

  public var debugDescription: String {
    return name
  }

  /** 
    Returns the last layer in this chain, or self if this is the last (or only)
    layer in the chain.
  */
  var lastLayer: Layer {
    var node = self
    while case let next? = node.next { node = next }
    return node
  }

  func summary(indent: Int = 0) -> (String, Int) {
    let i = String(repeating: " ", count: indent*2)
    let n = i + name.padding(toLength: 30 - indent*2, withPad: " ", startingAt: 0)
    let t = typeName.padding(toLength: 10, withPad: " ", startingAt: 0)
    let s = outputShape.debugDescription.padding(toLength: 16, withPad: " ", startingAt: 0)
    let p = weightCount + biasCount
    return (String(format: "%@ %@ %@ %d", n, t, s, p) + "\n", p)
  }

  /* Subclasses must implement these methods. */

  var typeName: String {
    fatalError("Subclass must implement this function")
  }

  func outputShape(for inputShape: DataShape) -> DataShape {
    fatalError("Subclass must implement this function")
  }

  func createCompute(device: MTLDevice, weights: ParameterData?, biases: ParameterData?) throws {
    // do nothing
  }

  func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    // do nothing
  }

  func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
    // do nothing
  }

  // If these return 0, we won't attempt to load any parameters for the layer.
  var weightCount: Int { return 0 }
  var biasCount: Int { return 0 }

  /* If a layer type has child layers (like Group) then it should implement
     the methods below. */

  func calculateOutputShapesForChildren(compiler: ModelCompiler) throws {
    // do nothing
  }

  func createComputeForChildren(compiler: ModelCompiler) throws {
    // do nothing
  }

  func encodeChildren(compiler: ModelCompiler,
                      commandBuffer: MTLCommandBuffer,
                      sourceImage: MPSImage?,
                      destinationImage: MPSImage?,
                      inflightIndex: Int) {
    // do nothing
  }
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

  public init(width: Int? = nil,
              height: Int? = nil,
              channels: Int? = nil,
              name: String = "") {

    super.init(name: name)
    self.inputShape = DataShape(width: width ?? -1,
                                height: height ?? -1,
                                channels: channels ?? -1)

    // We're not computing anything for this layer.
    needsEncoding = false
    needsDestinationImage = false

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
  This layer doesn't do anything. But it lets us write the following:
  
      let maybe: Layer
      if someCondition {
        maybe = SomeLayer() <-- SomeOtherLayer()
      } else {
        maybe = PassthroughLayer()
      }
      
      let model = Model()
                --> TheFirstLayer()
                --> maybe
                --> AnotherLayer()

  If `someCondition` is false, the PassthroughLayer is used, which effectively
  means nothing happens at that point. It's just a trick to make this kind of
  construct possible. The PassthroughLayer is actually removed from the model
  by the compiler.
*/
public class PassthroughLayer: Layer {
  public init() {
    super.init()
  }
}

/**
  Abstract base class for layers that encode a single MPSCNN kernel.
*/
public class MPSCNNLayer: Layer {
  var mpscnn: MPSCNNKernel!

  override func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage,
                       destinationImage: MPSImage) {
    mpscnn.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceImage,
                  destinationImage: destinationImage)
  }
}

/**
  Convolutional layer.
*/
public class Convolution: MPSCNNLayer {
  let kernel: (Int, Int)
  let channels: Int
  let stride: (Int, Int)
  let padding: Bool
  let filter: MPSCNNNeuron?

  /**
    Creates a convolution layer.
  
    - Parameters:
      - kernel: `(width, height)`
      - channels: Number of output channels.
      - stride: `(x, y)`
      - padding: If true, the output width and height are the same as the
        input width and height. (This uses zero padding.)
      - name: The name is used to load the layer's parameters.
  */
  public init(kernel: (Int, Int),
              channels: Int,
              stride: (Int, Int) = (1, 1),
              padding: Bool = true,
              filter: MPSCNNNeuron? = nil,
              name: String = "") {
    self.kernel = kernel
    self.channels = channels
    self.stride = stride
    self.padding = padding
    self.filter = filter
    super.init(name: name)
  }

  override var typeName: String {
    return "Conv"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    if padding {
      return DataShape(width: inputShape.width  / stride.0,
                      height: inputShape.height / stride.1,
                    channels: channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  channels)
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
                              biases: ParameterData?) throws {

    guard let weights = weights, let biases = biases else {
      throw ModelError.compileError(message: "missing parameters for layer '\(name)'")
    }

    let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: outputShape.channels,
                                           neuronFilter: filter)
    desc.strideInPixelsX = stride.0
    desc.strideInPixelsY = stride.1

    let conv = MPSCNNConvolution(device: device,
                                 convolutionDescriptor: desc,
                                 kernelWeights: weights.pointer,
                                 biasTerms: biases.pointer,
                                 flags: .none)

    conv.offset = calculatePadding(conv)
    conv.edgeMode = .zero
    conv.destinationFeatureChannelOffset = mergeOffset
    self.mpscnn = conv
  }

  func calculatePadding(_ conv: MPSCNNConvolution) -> MPSOffset {
    if padding {
      let padH = (outputShape.height - 1) * conv.strideInPixelsY + conv.kernelHeight - inputShape.height
      let padW = (outputShape.width  - 1) * conv.strideInPixelsX + conv.kernelWidth  - inputShape.width
      return MPSOffset(x: (conv.kernelWidth - padW)/2, y: (conv.kernelHeight - padH)/2, z: 0)
    } else {
      return MPSOffset(x: conv.kernelWidth/2, y: conv.kernelHeight/2, z: 0)
    }
  }
}

/**
  Abstract base class for max-pooling and average-pooling layers.
*/
public class Pooling: MPSCNNLayer {
  let kernel: (Int, Int)
  let stride: (Int, Int)
  let padding: Bool

  /**
    Creates a new pooling layer.
    
    - Parameters:
      - kernel: `(width, height)`
      - stride: `(x, y)`
      - padding: If true, the output width and height are the same as the
        input width and height. (This uses "clamp" padding.)
  */
  public init(kernel: (Int, Int),
              stride: (Int, Int),
              padding: Bool = false,
              name: String = "") {
    self.kernel = kernel
    self.stride = stride
    self.padding = padding
    super.init(name: name)
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    if padding {
      return DataShape(width: inputShape.width  / stride.0,
                      height: inputShape.height / stride.1,
                    channels: inputShape.channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  inputShape.channels)
    }
  }

  func calculatePadding(_ pool: MPSCNNPooling) -> MPSOffset {
    if padding {
      let padH = (outputShape.height - 1) * pool.strideInPixelsY + pool.kernelHeight - inputShape.height
      let padW = (outputShape.width  - 1) * pool.strideInPixelsX + pool.kernelWidth  - inputShape.width
      return MPSOffset(x: (pool.kernelWidth - padW)/2, y: (pool.kernelHeight - padH)/2, z: 0)
    } else {
      return MPSOffset(x: pool.kernelWidth/2, y: pool.kernelHeight/2, z: 0)
    }
  }

  override func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage,
                       destinationImage: MPSImage) {
    mpscnn.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceImage,
                  destinationImage: destinationImage)
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
                              biases: ParameterData?) throws {

    // FUTURE: Since pooling layers don't have parameters, it makes sense to
    // reuse them where possible. Put them into a dictionary using the kernel
    // size and stride as key.
    let pool = MPSCNNPoolingMax(device: device,
                                kernelWidth: kernel.0,
                                kernelHeight: kernel.1,
                                strideInPixelsX: stride.0,
                                strideInPixelsY: stride.1)

    pool.offset = calculatePadding(pool)
    pool.edgeMode = .clamp
    pool.destinationFeatureChannelOffset = mergeOffset
    self.mpscnn = pool
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
                              biases: ParameterData?) throws {

    // FUTURE: Also recycle these. See note in max-pool layer.
    let pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: kernel.0,
                                    kernelHeight: kernel.1,
                                    strideInPixelsX: stride.0,
                                    strideInPixelsY: stride.1)

    pool.offset = calculatePadding(pool)
    pool.edgeMode = .clamp
    pool.destinationFeatureChannelOffset = mergeOffset
    self.mpscnn = pool
  }
}

/**
  Global average-pooling layer
  
  This does the same thing as an AveragePooling layer with a kernel size equal
  to the input's spatial dimensions. If the input image is WxHxC, this averages
  across the width and height, and outputs a 1x1xC image.
*/
public class GlobalAveragePooling: MPSCNNLayer {
  public override init(name: String = "") {
    super.init(name: name)
  }

  override var typeName: String {
    return "GlbAvgPool"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: 1, height: 1, channels: inputShape.channels)
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) throws {

    let pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: inputShape.width,
                                    kernelHeight: inputShape.height,
                                    strideInPixelsX: inputShape.width,
                                    strideInPixelsY: inputShape.height)

    pool.offset = MPSOffset(x: inputShape.width/2, y: inputShape.height/2, z: 0)
    pool.edgeMode = .clamp
    pool.destinationFeatureChannelOffset = mergeOffset
    self.mpscnn = pool
  }
}

/**
  Fully-connected layer.
*/
public class Dense: MPSCNNLayer {
  let neurons: Int
  let filter: MPSCNNNeuron?

  /**
    Creates a fully-connected layer.
  
    - Parameters:
      - neurons: The number of neurons in this layer.
      - name: The name is used to load the layer's parameters.
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
                              biases: ParameterData?) throws {

    guard let weights = weights, let biases = biases else {
      throw ModelError.compileError(message: "missing parameters for layer '\(name)'")
    }

    // A fully-connected layer is a special version of a convolutional layer
    // where the kernel size is equal to the width/height of the input volume.
    // The output volume is 1x1xfanOut.
    let desc = MPSCNNConvolutionDescriptor(kernelWidth: inputShape.width,
                                           kernelHeight: inputShape.height,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: neurons,
                                           neuronFilter: filter)

    mpscnn = MPSCNNFullyConnected(device: device,
                                  convolutionDescriptor: desc,
                                  kernelWeights: weights.pointer,
                                  biasTerms: biases.pointer,
                                  flags: .none)

    mpscnn.destinationFeatureChannelOffset = mergeOffset
  }
}

/**
  Softmax layer.
*/
public class Softmax: MPSCNNLayer {
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
                              biases: ParameterData?) throws {
    mpscnn = MPSCNNSoftMax(device: device)
  }
}

/**
  Lets you use any MPSCNNNeuron as a layer of its own.
*/
public class Activation: MPSCNNLayer {
  public init(_ filter: MPSCNNNeuron, name: String = "") {
    super.init(name: name)
    self.mpscnn = filter
  }

  override var typeName: String {
    return "Activation"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }
}

/**
  Resizes the input texture to a specific size. The input is expected to have
  3 or 4 channels. Always outputs a 3-channel image.
*/
public class Resize: Layer {
  let width: Int
  let height: Int
  var lanczos: MPSImageLanczosScale!

  public init(width: Int, height: Int, name: String = "") {
    self.width = width
    self.height = height
    super.init(name: name)
    allowsIncompleteShape = true
    wantsTextures = true
  }

  override var typeName: String {
    return "Resize"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width, height: height, channels: 3)
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) throws {
    return lanczos = MPSImageLanczosScale(device: device)
  }

  override func encode(commandBuffer: MTLCommandBuffer,
                       sourceTexture: MTLTexture,
                       destinationTexture: MTLTexture) {
    lanczos.encode(commandBuffer: commandBuffer,
                   sourceTexture: sourceTexture,
                   destinationTexture: destinationTexture)
  }

  /**
    Crops the input image before it gets scaled down.

    The crop region is specified in input image coordinates.

    If you're always cropping the same region you can call this method right
    before or after compiling the model. If you're always cropping a different
    region (for example, using face detection on the input texture) then you
    should call this method right before you encode the model.
  */
  public func setCropRegion(x: Int, y: Int, width: Int, height: Int) {
    let scaleX = Double(outputShape.width) / Double(width)
    let scaleY = Double(outputShape.height) / Double(height)
    let translateX = Double(-x) * scaleX
    let translateY = Double(-y) * scaleY
    var transform = MPSScaleTransform(scaleX: scaleX,
                                      scaleY: scaleY,
                                      translateX: translateX,
                                      translateY: translateY)

    withUnsafePointer(to: &transform) { ptr in
      lanczos.scaleTransform = ptr
    }
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

  override func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage,
                       destinationImage: MPSImage) {
    custom.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceImage,
                  destinationImage: destinationImage)
  }
}

/**
  A group runs several layers in parallel. All layers receive the same input,
  and their output is depth-concatenated. Useful for making Inception towers.
  
  - Note: If you nest a Group inside a Group, then the nested Group will render
    into the outer Group's destinationImage -- it does not get its own image.
*/
public class Group: Layer {
  let children: [Layer]

  public init(_ layers: [Layer], name: String = "") {
    self.children = layers
    super.init(name: name)
    needsEncoding = false
  }

  override var typeName: String {
    return "Group"
  }

  override func calculateOutputShapesForChildren(compiler: ModelCompiler) throws {
    for layer in children {
      layer.inputShape = inputShape
      try compiler.calculateOutputShapes(for: layer)
    }
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
    var maxWidth = 0
    var maxHeight = 0
    var channels = 0

    for layer in children {
      let lastLayer = layer.lastLayer

      // The last layers of each of the sequences will be merged into
      // one big image (which is the image allocated for the group).
      lastLayer.mergeOffset = channels
      lastLayer.needsDestinationImage = false

      maxWidth = max(maxWidth, lastLayer.outputShape.width)
      maxHeight = max(maxHeight, lastLayer.outputShape.height)
      channels += lastLayer.outputShape.channels
    }
    return DataShape(width: maxWidth, height: maxHeight, channels: channels)
  }

  override func createComputeForChildren(compiler: ModelCompiler) throws {
    for layer in children {
      var ptr: Layer? = layer
      while ptr != nil, let node = ptr {
        // If this is a nested Group, then its children's merge offsets are
        // relative to that of the group itself.
        if node.next == nil {
          node.mergeOffset += self.mergeOffset
        }

        try compiler.createComputeForLayer(node)
        ptr = node.next
      }
    }
  }

  override func encodeChildren(compiler: ModelCompiler,
                               commandBuffer: MTLCommandBuffer,
                               sourceImage: MPSImage?,
                               destinationImage: MPSImage?,
                               inflightIndex: Int) {

    // This allows us to read multiple times from a temporary image.
    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount = children.count
    }

    for layer in children {
      var image = sourceImage
      var ptr: Layer? = layer
      while ptr != nil, let node = ptr {
        image = compiler.encode(commandBuffer: commandBuffer,
                                layer: node,
                                sourceImage: image,
                                destinationImage: destinationImage,
                                inflightIndex: inflightIndex)
        ptr = node.next
      }
    }
  }

  override func summary(indent: Int) -> (String, Int) {
    var (text, params) = super.summary(indent: indent)

    for layer in children {
      var ptr: Layer? = layer
      while ptr != nil, let node = ptr {
        let (t, p) = node.summary(indent: indent + 1)
        text += t
        params += p
        ptr = node.next
      }
    }
    return (text, params)
  }
}

public class DepthwiseConvolution: Layer {
  let kernel: (Int, Int)
  let stride: (Int, Int)
  let useReLU: Bool
  var compute: DepthwiseConvolutionKernel!

  /**
    Creates a depth-wise convolution layer.
  
    - Parameters:
      - kernel: `(width, height)`
      - stride: `(x, y)`
      - useReLU: Whether to apply a ReLU directly in the shader. You can also
        add `Activation(relu)` behind this layer instead.
      - name: The name is used to load the layer's parameters.
  */
  public init(kernel: (Int, Int),
              stride: (Int, Int) = (1, 1),
              useReLU: Bool = true,
              name: String = "") {
    self.kernel = kernel
    self.stride = stride
    self.useReLU = useReLU
    super.init(name: name)
  }

  override var typeName: String {
    return "DepthwConv"
  }

  override func outputShape(for inputShape: DataShape) -> DataShape {
      return DataShape(width: inputShape.width  / stride.0,
                      height: inputShape.height / stride.1,
                    channels: inputShape.channels)
  }

  override var weightCount: Int {
    return inputShape.channels * kernel.1 * kernel.0
  }

  override var biasCount: Int {
    return 0
  }

  override func createCompute(device: MTLDevice,
                              weights: ParameterData?,
                              biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing parameters for layer '\(name)'")
    }

    compute = DepthwiseConvolutionKernel(device: device,
                                         kernelWidth: kernel.0,
                                         kernelHeight: kernel.1,
                                         featureChannels: inputShape.channels,
                                         strideInPixelsX: stride.0,
                                         strideInPixelsY: stride.1,
                                         channelMultiplier: 1,
                                         relu: useReLU,
                                         kernelWeights: weights.pointer)
  }

  override func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage,
                       destinationImage: MPSImage) {
    compute.encode(commandBuffer: commandBuffer,
                   sourceImage: sourceImage,
                   destinationImage: destinationImage)
  }
}

public class PointwiseConvolution: Convolution {
  /**
    Creates a point-wise convolution layer, which is really the same as a 
    convolutional layer with a 1x1 kernel.
  */
  public init(channels: Int,
              stride: (Int, Int) = (1, 1),
              filter: MPSCNNNeuron? = nil,
              name: String = "") {
    super.init(kernel: (1, 1), channels: channels, filter: filter, name: name)
  }

  override var typeName: String {
    return "PointwConv"
  }
}
