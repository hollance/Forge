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

public enum PaddingType {
  case same    // add zero padding
  case valid   // don't add padding
}

func offsetForConvolution(padding: PaddingType,
                          sourceWidth: Int,
                          sourceHeight: Int,
                          destinationWidth: Int,
                          destinationHeight: Int,
                          kernelWidth: Int,
                          kernelHeight: Int,
                          strideInPixelsX: Int,
                          strideInPixelsY: Int) -> MPSOffset {
  if padding == .same {
    let padH = (destinationHeight - 1) * strideInPixelsY + kernelHeight - sourceHeight
    let padW = (destinationWidth  - 1) * strideInPixelsX + kernelWidth  - sourceWidth
    return MPSOffset(x: (kernelWidth - padW)/2, y: (kernelHeight - padH)/2, z: 0)
  } else {
    return MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
  }
}

func offsetForPooling(padding: PaddingType,
                      sourceWidth: Int,
                      sourceHeight: Int,
                      kernelWidth: Int,
                      kernelHeight: Int,
                      strideInPixelsX: Int,
                      strideInPixelsY: Int) -> MPSOffset {
  if padding == .same {
    var offset = MPSOffset(x: 0, y: 0, z: 0)
    if kernelWidth % 2 == 0 {
      offset.x += (((sourceWidth - 1) % strideInPixelsX) / 2) + 1
    } else {
      offset.x += (((sourceWidth - 1) % strideInPixelsX) + 1) / 2
    }
    if kernelHeight % 2 == 0 {
      offset.y += (((sourceHeight - 1) % strideInPixelsY) / 2) + 1
    } else {
      offset.y += (((sourceHeight - 1) % strideInPixelsY) + 1) / 2
    }
    return offset
  } else {
    return MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
  }
}

/**
  The abstract base class for all layers. You should not create instances of
  this class directly.
*/
open class Layer {
  internal(set) public var name: String

  // Most layers take MPSImages as input but for Resize it's more optimal to
  // work directly on the input texture. That saves making an MPSImage object.
  // Probably a premature optimization. ;-)
  internal(set) public var wantsTextures: Bool

  // Most layers require that the complete shape of the input tensor is known.
  // However, some layers (such as Resize and Custom) can handle inputs of any
  // size. If your first layer is a type that must know the size (Convolution)
  // then you need to specify that size to the Input tensor.
  internal(set) public var allowsIncompleteShape: Bool

  /* Whether this layer uses bias terms in addition to weights. */
  internal(set) public var useBias: Bool

  // The same layer can be used by multiple tensors, but we should only create
  // its compute just once. Reusing layers is mostly useful for things like
  // pooling, which don't take parameters.
  var createdCompute = false

  // The parameter count shown in the summary. (Filled in by the compiler.)
  var paramCount = 0

  public init(name: String = "",
              useBias: Bool = true,
              wantsTextures: Bool = false,
              allowsIncompleteShape: Bool = false) {
    self.name = name
    self.useBias = useBias
    self.wantsTextures = wantsTextures
    self.allowsIncompleteShape = allowsIncompleteShape
  }

  /* Subclasses must implement these methods. */

  open var typeName: String {
    fatalError("Subclass must implement this function")
  }

  open func outputShape(for inputShape: DataShape) -> DataShape {
    fatalError("Subclass must implement this function")
  }

  open func createCompute(device: MTLDevice,
                          inputShape: DataShape,
                          outputShape: DataShape,
                          weights: ParameterData?,
                          biases: ParameterData?) throws {
    // do nothing
  }

  open func encode(commandBuffer: MTLCommandBuffer,
                   sourceTensor: Tensor,
                   destinationTensor: Tensor) {
    // Note: sourceTensor.image and destinationTensor.image are guaranteed
    // to be non-nil at this point, so it's OK to force-unwrap them.
  }

  open func encode(commandBuffer: MTLCommandBuffer,
                   sourceTensor: Tensor,
                   sourceTexture: MTLTexture,
                   destinationTensor: Tensor) {
    // This is a special-case method for layers that prefer to work with
    // textures rather than MPSImages. The output will always be a texture
    // from an MPSImage but this not necessarily true for the input texture.
  }

  open func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return 0
  }

  open func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return 0
  }
}

extension Layer: CustomDebugStringConvertible {
  public var debugDescription: String {
    return name
  }
}

/**
  Abstract base class for layers that encode a single MPSCNN kernel.
*/
public class MPSCNNLayer: Layer {
  var mpscnn: MPSCNNKernel!

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    // FUTURE: For a residual connection, where we may want to read from the
    // destination image (and write to that same destination image), we would
    // set mpscnn.offset and clipRect here using sourceTensor's channel offset.

    mpscnn.destinationFeatureChannelOffset = destinationTensor.destinationChannelOffset

    mpscnn.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceTensor.image!,
                  destinationImage: destinationTensor.image!)
  }
}

/**
  Convolutional layer.
*/
public class Convolution: MPSCNNLayer {
  let kernel: (Int, Int)
  let channels: Int
  let stride: (Int, Int)
  let padding: PaddingType
  let activation: MPSCNNNeuron?
  var conv: MPSCNNConvolution!

  /**
    Creates a convolution layer.
  
    - Parameters:
      - kernel: `(width, height)`
      - channels: Number of output channels.
      - stride: `(x, y)`
      - padding: If .same, the output width and height are the same as the
        input width and height. (This uses zero padding.)
      - useBias: whether this layer uses bias terms in addition to the weights
      - name: The name is used to load the layer's parameters.
  */
  public init(kernel: (Int, Int),
              channels: Int,
              stride: (Int, Int) = (1, 1),
              padding: PaddingType = .same,
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    self.kernel = kernel
    self.channels = channels
    self.stride = stride
    self.padding = padding
    self.activation = activation
    super.init(name: name, useBias: useBias)
  }

  override public var typeName: String {
    return "Conv"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    if padding == .same {
      return DataShape(width: (inputShape.width - 1)  / stride.0 + 1,
                      height: (inputShape.height - 1) / stride.1 + 1,
                    channels: channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  channels)
    }
  }

  override public func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return inputShape.channels * kernel.1 * kernel.0 * outputShape.channels
  }

  override public func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return useBias ? outputShape.channels : 0
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing weights for layer '\(name)'")
    }

    let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: outputShape.channels,
                                           neuronFilter: activation)
    desc.strideInPixelsX = stride.0
    desc.strideInPixelsY = stride.1

    if useBias {
      guard let biases = biases else {
        throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
      }

      conv = MPSCNNConvolution(device: device,
                               convolutionDescriptor: desc,
                               kernelWeights: weights.pointer,
                               biasTerms: biases.pointer,
                               flags: .none)
    } else {
      conv = MPSCNNConvolution(device: device,
                               convolutionDescriptor: desc,
                               kernelWeights: weights.pointer,
                               biasTerms: nil,
                               flags: .none)
    }
    conv.edgeMode = .zero
    mpscnn = conv
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    // We compute the padding at encode-time, so that this layer can be
    // reused on tensors of different sizes. Note that the input and output
    // depth must not vary, only the width and height may be different.
    conv.offset = offsetForConvolution(padding: padding,
                                       sourceWidth: sourceTensor.shape.width,
                                       sourceHeight: sourceTensor.shape.height,
                                       destinationWidth: destinationTensor.shape.width,
                                       destinationHeight: destinationTensor.shape.height,
                                       kernelWidth: kernel.0,
                                       kernelHeight: kernel.1,
                                       strideInPixelsX: stride.0,
                                       strideInPixelsY: stride.1)

    super.encode(commandBuffer: commandBuffer,
                 sourceTensor: sourceTensor,
                 destinationTensor: destinationTensor)
  }
}

/**
  Abstract base class for max-pooling and average-pooling layers.
*/
public class Pooling: MPSCNNLayer {
  let kernel: (Int, Int)
  let stride: (Int, Int)
  let padding: PaddingType
  let edgeMode: MPSImageEdgeMode
  var pool: MPSCNNPooling!

  /**
    Creates a new pooling layer.
    
    - Parameters:
      - kernel: `(width, height)`
      - stride: `(x, y)`
      - padding: Whether to add padding around the input image. (This uses 
                 "clamp" padding.)
  */
  public init(kernel: (Int, Int),
              stride: (Int, Int),
              padding: PaddingType = .valid,
              edgeMode: MPSImageEdgeMode = .clamp,
              name: String = "") {
    self.kernel = kernel
    self.stride = stride
    self.padding = padding
    self.edgeMode = edgeMode
    super.init(name: name)
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    if padding == .same {
      return DataShape(width: (inputShape.width - 1)  / stride.0 + 1,
                      height: (inputShape.height - 1) / stride.1 + 1,
                    channels: inputShape.channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  inputShape.channels)
    }
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    // We compute the padding at encode-time, so that this layer can be
    // reused on tensors of different sizes.
    pool.offset = offsetForPooling(padding: padding,
                                   sourceWidth: sourceTensor.shape.width,
                                   sourceHeight: sourceTensor.shape.height,
                                   kernelWidth: kernel.0,
                                   kernelHeight: kernel.1,
                                   strideInPixelsX: stride.0,
                                   strideInPixelsY: stride.1)

    super.encode(commandBuffer: commandBuffer,
                 sourceTensor: sourceTensor,
                 destinationTensor: destinationTensor)
  }
}

/**
  Max-pooling layer.
*/
public class MaxPooling: Pooling {
  override public var typeName: String {
    return "MaxPool"
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    pool = MPSCNNPoolingMax(device: device,
                            kernelWidth: kernel.0,
                            kernelHeight: kernel.1,
                            strideInPixelsX: stride.0,
                            strideInPixelsY: stride.1)
    pool.edgeMode = edgeMode
    mpscnn = pool
  }
}

/**
  Average-pooling layer.
*/
public class AveragePooling: Pooling {
  override public var typeName: String {
    return "AvgPool"
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    pool = MPSCNNPoolingAverage(device: device,
                                kernelWidth: kernel.0,
                                kernelHeight: kernel.1,
                                strideInPixelsX: stride.0,
                                strideInPixelsY: stride.1)
    pool.edgeMode = edgeMode
    mpscnn = pool
  }
}

/**
  Global average-pooling layer
  
  This does the same thing as an AveragePooling layer with a kernel size equal
  to the input's spatial dimensions. If the input image is WxHxC, this averages
  across the width and height, and outputs a 1x1xC image.
*/
public class GlobalAveragePooling: MPSCNNLayer {
  override public var typeName: String {
    return "GlbAvgPool"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: 1, height: 1, channels: inputShape.channels)
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    let pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: inputShape.width,
                                    kernelHeight: inputShape.height,
                                    strideInPixelsX: inputShape.width,
                                    strideInPixelsY: inputShape.height)

    pool.offset = MPSOffset(x: inputShape.width/2, y: inputShape.height/2, z: 0)
    pool.edgeMode = .clamp
    self.mpscnn = pool
  }
}

/**
  Fully-connected layer.
*/
public class Dense: MPSCNNLayer {
  let neurons: Int
  let activation: MPSCNNNeuron?

  /**
    Creates a fully-connected layer.
  
    - Parameters:
      - neurons: The number of neurons in this layer.
      - name: The name is used to load the layer's parameters.
  */
  public init(neurons: Int,
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    self.neurons = neurons
    self.activation = activation
    super.init(name: name, useBias: useBias)
  }

  override public var typeName: String {
    return "Dense"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: 1, height: 1, channels: neurons)
  }

  override public func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return inputShape.channels * inputShape.height * inputShape.width * neurons
  }

  override public func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return useBias ? neurons : 0
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing weights for layer '\(name)'")
    }

    // A fully-connected layer is a special version of a convolutional layer
    // where the kernel size is equal to the width/height of the input volume.
    // The output volume is 1x1xfanOut.
    let desc = MPSCNNConvolutionDescriptor(kernelWidth: inputShape.width,
                                           kernelHeight: inputShape.height,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: neurons,
                                           neuronFilter: activation)

    // NOTE: With optimizations enabled, the biases object gets deallocated too
    // soon, even though we still have a strong reference to it. The following
    // workaround fixes this, since `guard let` creates a new strong reference.
    if useBias {
      guard let biases = biases else {
        throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
      }

      mpscnn = MPSCNNFullyConnected(device: device,
                                    convolutionDescriptor: desc,
                                    kernelWeights: weights.pointer,
                                    biasTerms: biases.pointer,
                                    flags: .none)
    } else {
      mpscnn = MPSCNNFullyConnected(device: device,
                                    convolutionDescriptor: desc,
                                    kernelWeights: weights.pointer,
                                    biasTerms: nil,
                                    flags: .none)
    }
  }
}

/**
  Softmax layer.
*/
public class Softmax: MPSCNNLayer {
  override public var typeName: String {
    return "Softmax"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {
    mpscnn = MPSCNNSoftMax(device: device)
  }
}

/**
  Lets you use any MPSCNNNeuron as a layer of its own.
*/
public class Activation: MPSCNNLayer {
  public init(_ activation: MPSCNNNeuron, name: String = "") {
    super.init(name: name)
    self.mpscnn = activation
  }

  override public var typeName: String {
    return "Activation"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
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

  override public var typeName: String {
    return "Resize"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width, height: height, channels: 3)
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {
    return lanczos = MPSImageLanczosScale(device: device)
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              sourceTexture: MTLTexture,
                              destinationTensor: Tensor) {
    lanczos.encode(commandBuffer: commandBuffer,
                   sourceTexture: sourceTexture,
                   destinationTexture: destinationTensor.image!.texture)
  }

  /**
    Crops the input image before it gets scaled down.

    The crop region is specified in input image coordinates.

    If you're always cropping the same region you can call this method right
    before or after compiling the model. If you're always cropping a different
    region (for example, using face detection on the input texture) then you
    should call this method right before you encode the model.
  */
  public func setCropRect(x: Double, y: Double, width: Double, height: Double) {
    let scaleX = Double(self.width) / width
    let scaleY = Double(self.height) / height
    let translateX = -x * scaleX
    let translateY = -y * scaleY
    var transform = MPSScaleTransform(scaleX: scaleX,
                                      scaleY: scaleY,
                                      translateX: translateX,
                                      translateY: translateY)

    withUnsafePointer(to: &transform) { ptr in
      lanczos.scaleTransform = ptr
    }
  }

  public func setCropRect(_ rect: CGRect) {
    setCropRect(x: Double(rect.origin.x),
                y: Double(rect.origin.y),
                width: Double(rect.width),
                height: Double(rect.height))
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

  override public var typeName: String {
    return "Custom"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width ?? inputShape.width,
                     height: height ?? inputShape.height,
                     channels: channels ?? inputShape.channels)
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {
    custom.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceTensor.image!,
                  destinationImage: destinationTensor.image!)
  }
}

public class DepthwiseConvolution: Layer {
  let kernel: (Int, Int)
  let stride: (Int, Int)
  let activation: MPSCNNNeuron?
  var compute: Any!

  /**
    Creates a depth-wise convolution layer.
    
    Currently only supports .same padding.
  
    - Parameters:
      - kernel: `(width, height)`
      - stride: `(x, y)`
      - useReLU: Whether to apply a ReLU directly in the shader. You can also
        add `Activation(relu)` behind this layer instead.
      - name: The name is used to load the layer's parameters.
  */
  public init(kernel: (Int, Int),
              stride: (Int, Int) = (1, 1),
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    self.kernel = kernel
    self.stride = stride
    self.activation = activation
    super.init(name: name, useBias: useBias)
  }

  override public var typeName: String {
    return "DepthwConv"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
      return DataShape(width: (inputShape.width - 1)  / stride.0 + 1,
                      height: (inputShape.height - 1) / stride.1 + 1,
                    channels: inputShape.channels)
  }

  override public func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return inputShape.channels * kernel.1 * kernel.0
  }

  public override func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return useBias ? outputShape.channels : 0
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing weights for layer '\(name)'")
    }

    if #available(iOS 11.0, *) {
      let desc = MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: kernel.0,
                                                      kernelHeight: kernel.1,
                                                      inputFeatureChannels: inputShape.channels,
                                                      outputFeatureChannels: inputShape.channels,
                                                      neuronFilter: activation)
      desc.strideInPixelsX = stride.0
      desc.strideInPixelsY = stride.1

      if useBias {
        guard let biases = biases else {
          throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
        }

        compute = MPSCNNConvolution(device: device,
                                    convolutionDescriptor: desc,
                                    kernelWeights: weights.pointer,
                                    biasTerms: biases.pointer,
                                    flags: .none)
      } else {
        compute = MPSCNNConvolution(device: device,
                                    convolutionDescriptor: desc,
                                    kernelWeights: weights.pointer,
                                    biasTerms: nil,
                                    flags: .none)
      }
      (compute as! MPSCNNConvolution).edgeMode = .zero

    } else {
      if useBias {
        guard let biases = biases else {
          throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
        }

        compute = DepthwiseConvolutionKernel(device: device,
                                             kernelWidth: kernel.0,
                                             kernelHeight: kernel.1,
                                             featureChannels: inputShape.channels,
                                             strideInPixelsX: stride.0,
                                             strideInPixelsY: stride.1,
                                             channelMultiplier: 1,
                                             neuronFilter: activation,
                                             kernelWeights: weights.pointer,
                                             biasTerms: biases.pointer)
      } else {
        compute = DepthwiseConvolutionKernel(device: device,
                                             kernelWidth: kernel.0,
                                             kernelHeight: kernel.1,
                                             featureChannels: inputShape.channels,
                                             strideInPixelsX: stride.0,
                                             strideInPixelsY: stride.1,
                                             channelMultiplier: 1,
                                             neuronFilter: activation,
                                             kernelWeights: weights.pointer,
                                             biasTerms: nil)
      }
    }
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {
    let offset = offsetForConvolution(padding: .same,
                                      sourceWidth: sourceTensor.shape.width,
                                      sourceHeight: sourceTensor.shape.height,
                                      destinationWidth: destinationTensor.shape.width,
                                      destinationHeight: destinationTensor.shape.height,
                                      kernelWidth: kernel.0,
                                      kernelHeight: kernel.1,
                                      strideInPixelsX: stride.0,
                                      strideInPixelsY: stride.1)

    if let compute = compute as? MPSCNNConvolution {
      compute.offset = offset
      compute.encode(commandBuffer: commandBuffer,
                     sourceImage: sourceTensor.image!,
                     destinationImage: destinationTensor.image!)
    } else if let compute = compute as? DepthwiseConvolutionKernel {
      compute.offset = offset
      compute.encode(commandBuffer: commandBuffer,
                     sourceImage: sourceTensor.image!,
                     destinationImage: destinationTensor.image!)
    }
  }
}

public class PointwiseConvolution: Convolution {
  /**
    Creates a point-wise convolution layer, which is really the same as a 
    convolutional layer with a 1x1 kernel.
  */
  public init(channels: Int,
              stride: (Int, Int) = (1, 1),
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    super.init(kernel: (1, 1), channels: channels, activation: activation,
               useBias: useBias, name: name)
  }

  override public var typeName: String {
    return "PointwConv"
  }
}
