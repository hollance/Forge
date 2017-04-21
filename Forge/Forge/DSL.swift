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
import QuartzCore

/*
  A simple DSL for building neural networks without the boilerplate.
*/

precedencegroup ChainPrecedence {
  associativity: right
  higherThan: MultiplicationPrecedence
}

infix operator => : ChainPrecedence

public func => (lhs: Layer, rhs: Layer) -> Layer {
  // This operator goes from right to left. To make this work correctly, we
  // always add the new layer to the end of the list. This also lets you use
  // parens, e.g. `model => (layer1 => layer2) => layer3`.
  var node = lhs
  while case let next? = node.next { node = next }
  node.next = rhs
  return lhs
}

public func => (lhs: Model, rhs: Layer) -> Model {
  // FUTURE: writing `model => layer` adds the first layer to a dictionary,
  // using the layer's name as the key. This lets you add multiple inputs to
  // the model by writing `model => layer` multiple times. When predicting,
  // you feed a dictionary of textures into predict().
  lhs.firstLayer = rhs
  return lhs
}

enum Compute {
  case mpscnn(MPSCNNKernel)
  case mps(MPSUnaryImageKernel)
  case custom(CustomKernel)
}

/**
  Describes the dimensions of the data as it flows through the neural net.

  Because Input can accept a texture of unknown size, we use -1 to indicate
  that a dimension is not known yet. (Don't want to use optionals for this,
  since most of the time the dimensions *will* be known and unwrapping just
  makes the code uglier.)
*/
struct DataShape: Hashable, CustomDebugStringConvertible {
  let width: Int
  let height: Int
  let channels: Int

  init(width: Int = -1, height: Int = -1, channels: Int = -1) {
    self.width = width
    self.height = height
    self.channels = channels
  }

  var isFullySpecified: Bool {
    return width != -1 && height != -1 && channels != -1
  }

  // Needs to be hashable because we'll create a cache of MPSImageDescriptor
  // objects. The DataShape is the key they're stored under.
  var hashValue: Int {
    return width + height*1000 + channels*1000*1000
  }

  func createImageDescriptor() -> MPSImageDescriptor {
    assert(isFullySpecified)
    return MPSImageDescriptor(channelFormat: .float16, width: width,
                              height: height, featureChannels: channels)
  }

  var debugDescription: String {
    var dims: [String] = []
    if width    != -1 { dims.append("\(width)")    } else { dims.append("?") }
    if height   != -1 { dims.append("\(height)")   } else { dims.append("?") }
    if channels != -1 { dims.append("\(channels)") } else { dims.append("?") }
    return "(" + dims.joined(separator: ", ") + ")"
  }
}

func == (lhs: DataShape, rhs: DataShape) -> Bool {
  return lhs.width    == rhs.width
      && lhs.height   == rhs.height
      && lhs.channels == rhs.channels
}

/**
  The top-level object for the neural network.
  
  You first add layers to the model using `=>` and then call `compile()` to 
  construct all the Metal objects.
*/
public class Model {
  var firstLayer: Layer?

  var compiled = false
  var numLayers = 0
  var imageDescriptors: [DataShape: MPSImageDescriptor] = [:]
  var imageDescriptorList: [MPSImageDescriptor] = []
  var outputImages: [String: [MPSImage]] = [:]
  var lastLayerName = ""
  var firstLayerEatsTexture = false

  public init() { }

  /**
    Creates all the MPSCNN objects for this graph.
  */
  public func compile(device: MTLDevice, inflightBuffers: Int) -> Bool {
    if compiled {
      print("Compile error: graph has already been compiled")
      return false
    }

    guard let layer = firstLayer else {
      print("Compile error: model needs at least one layer")
      return false
    }

    let startTime = CACurrentMediaTime()

    var firstComputeLayer = true
    var ptr: Layer? = layer
    while ptr != nil, let node = ptr {
      numLayers += 1

      if node.name.isEmpty {
        node.name = "__\(node.typeName)_\(numLayers)__"
      }

      if !node.inputShape.isFullySpecified && !node.allowsIncompleteShape {
        print("Compile error: input shape \(node.inputShape) for node '\(node)' has unknown dimensions")
        return false
      }

      let shape = node.outputShape(for: node.inputShape)
      node.outputShape = shape

      if shape.isFullySpecified {
        if imageDescriptors[shape] == nil {
          imageDescriptors[shape] = shape.createImageDescriptor()
        }

        if let compute = node.createCompute(device: device) {
          node.compute = compute

          // Does the first layer take a MTLTexture or an MPSImage?
          if firstComputeLayer {
            if case .mps = compute { firstLayerEatsTexture = true }
            firstComputeLayer = false
          }
        } else {
          print("Compile error: could not create compute object for node '\(node)'")
          return false
        }
      }

      ptr = node.next
      ptr?.inputShape = shape

      // Make an MPSImage for the last node or for any node that asks for it.
      // We keep track of these in a dictionary.
      if ptr == nil || !node.temporaryImage {
        guard let imgDesc = imageDescriptors[node.outputShape] else {
          fatalError("Error: could not find image descriptor for shape \(node.outputShape)")
        }

        // Since the GPU can be working on several inputs at once, we need to
        // allocate multiple output images.
        var array: [MPSImage] = []
        for _ in 0..<inflightBuffers {
          array.append(MPSImage(device: device, imageDescriptor: imgDesc))
        }
        outputImages[node.name] = array

        // Keep track of the last layer so we can easily find its MPSImage.
        lastLayerName = node.name
      }
    }

    imageDescriptorList = Array(imageDescriptors.values)

    let elapsed = CACurrentMediaTime() - startTime
    print("Compiling took \(elapsed) seconds")

    compiled = true
    return true
  }

  /**
    Creates the GPU commands for a forward pass of the neural network.

    - FUTURE: Takes a dict with textures for multiple inputs. The inputs are
      identified by name.
              
    - FUTURE: We want encoding to be as fast as possible, so we could move all
      the below logic into compile(), which then outputs a list of commands. 
      Example:
          1: use MPS kernel X with the input texture, output to temp image Y
          2: use MPSCNN kernel X with new temp image of shape Y
          2: use MPSCNN kernel X with given MPSImage Y
          3: use custom kernel X with new temp image of shape Y
          4: ...
      Then we just go through the list and switch on the command.
  */
  public func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int) {
    //let startTime = CACurrentMediaTime()

    MPSTemporaryImage.prefetchStorage(with: commandBuffer,
                                      imageDescriptorList: imageDescriptorList)

    var sourceImage: MPSImage!
    if !firstLayerEatsTexture {
      sourceImage = MPSImage(texture: texture, featureChannels: 3)
    }

    var ptr: Layer? = firstLayer
    while ptr != nil, let node = ptr {
      if let compute = node.compute {
        // Allocate the MPSTemporaryImage to hold the output for this layer.
        // If the node has a real MPSImage instead, then use that.
        let destinationImage: MPSImage
        if let images = outputImages[node.name] {
          destinationImage = images[inflightIndex]
        } else {
          destinationImage = MPSTemporaryImage(commandBuffer: commandBuffer,
                                               imageDescriptor: imageDescriptors[node.outputShape]!)
        }

        switch compute {
        case .mpscnn(let kernel):
          kernel.encode(commandBuffer: commandBuffer,
                        sourceImage: sourceImage,
                        destinationImage: destinationImage)

        case .mps(let kernel):
          let inputTexture: MTLTexture
          if sourceImage != nil {
            inputTexture = sourceImage.texture
          } else {
            inputTexture = texture   // valid only for first layer
          }
          kernel.encode(commandBuffer: commandBuffer,
                        sourceTexture: inputTexture,
                        destinationTexture: destinationImage.texture)

        case .custom(let kernel):
          kernel.encode(commandBuffer: commandBuffer,
                        sourceImage: sourceImage,
                        destinationImage: destinationImage)

          // Let Metal know the temporary image can be recycled.
          if let tmp = sourceImage as? MPSTemporaryImage {
            tmp.readCount -= 1
          }
        }
        sourceImage = destinationImage
      }
      ptr = node.next
    }

    //let elapsed = CACurrentMediaTime() - startTime
    //print("Encoding took \(elapsed) seconds")
  }

  /**
    Returns a summary of all the layers in the model, including their types,
    shapes, and number of parameters. Useful for making sure your model is
    correct.
  */
  public func summary() -> String {
    precondition(compiled)

    var s = ""
    s += "Layer                Type       Output Shape     Parameters\n"
    s += "-----------------------------------------------------------\n"

    var totalParams = 0
    var ptr: Layer? = firstLayer
    while ptr != nil, let node = ptr {
      let name = node.name.padding(toLength: 20, withPad: " ", startingAt: 0)
      let type = node.typeName.padding(toLength: 10, withPad: " ", startingAt: 0)
      let shape = node.outputShape.debugDescription.padding(toLength: 16, withPad: " ", startingAt: 0)
      let params = node.weightCount + node.biasCount
      totalParams += params
      s += String(format: "%@ %@ %@ %d", name, type, shape, params) + "\n"
      ptr = node.next
    }

    s += "-----------------------------------------------------------\n"
    s += "Number of layers: \(numLayers)\n"
    s += "Total parameters: \(totalParams)"
    return s
  }

  /**
    Find a layer by name. You can use this before compiling to change the
    properties of a layer (only some properties can be changed).
  */
  public subscript(name: String) -> Layer? {
    var ptr: Layer? = firstLayer
    while ptr != nil, let node = ptr {
      if node.name == name { return node }
      ptr = node.next
    }
    return nil
  }

  /** 
    Returns the output from the given layer. This layer must have its
    temporaryImage property set to false!
  */
  public func images(forLayer name: String) -> [MPSImage] {
    return outputImages[name] ?? []
  }

  public func image(forLayer name: String, inflightIndex: Int) -> MPSImage {
    return images(forLayer: name)[inflightIndex]
  }

  /** Returns the output from the last layer in the model. */
  public func outputImage(inflightIndex: Int) -> MPSImage {
    return outputImages[lastLayerName]![inflightIndex]
  }
}

/**
  The abstract base class for all layers. You cannot create instances of this 
  class directly.
  
  When you create a model, you're actually building a graph of these Layer
  objects.
*/
public class Layer: CustomDebugStringConvertible {
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

  func createCompute(device: MTLDevice) -> Compute? {
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

  override func createCompute(device: MTLDevice) -> Compute? {
    let sizeWeights = weightCount * MemoryLayout<Float>.stride
    let sizeBias = biasCount * MemoryLayout<Float>.stride

    guard let weightsPath = Bundle.main.path(forResource: name + "_W", ofType: "bin"),
          let weightsData = Parameters(path: weightsPath, fileSize: sizeWeights),
          let biasPath = Bundle.main.path(forResource: name + "_b", ofType: "bin"),
          let biasData = Parameters(path: biasPath, fileSize: sizeBias) else {
      print("Error loading network parameters '\(name)'")
      return nil
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
                                  kernelWeights: weightsData.pointer,
                                  biasTerms: biasData.pointer,
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

  override func createCompute(device: MTLDevice) -> Compute? {
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

  override func createCompute(device: MTLDevice) -> Compute? {
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

  override func createCompute(device: MTLDevice) -> Compute? {
    let sizeWeights = weightCount * MemoryLayout<Float>.stride
    let sizeBias = biasCount * MemoryLayout<Float>.stride

    // FUTURE: create a callback that gets invoked on compile, that gives
    // the client the chance to load the parameters in whatever way they
    // want. During compile(), we'll sort the layers so that the largest is
    // loaded first.
    guard let weightsPath = Bundle.main.path(forResource: name + "_W", ofType: "bin"),
          let weightsData = Parameters(path: weightsPath, fileSize: sizeWeights),
          let biasPath = Bundle.main.path(forResource: name + "_b", ofType: "bin"),
          let biasData = Parameters(path: biasPath, fileSize: sizeBias) else {
      print("Error loading network parameters '\(name)'")
      return nil
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
                                     kernelWeights: weightsData.pointer,
                                     biasTerms: biasData.pointer,
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

  override func createCompute(device: MTLDevice) -> Compute? {
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

  override func createCompute(device: MTLDevice) -> Compute? {
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

  override func createCompute(device: MTLDevice) -> Compute? {
    return Compute.mps(MPSImageLanczosScale(device: device))
  }
}

/**
  The Custom layer type accepts any object that conforms to this protocol.
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

  override func createCompute(device: MTLDevice) -> Compute? {
    return Compute.custom(custom)
  }
}
