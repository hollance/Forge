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

infix operator --> : ChainPrecedence

public func --> (lhs: Layer, rhs: Layer) -> Layer {
  // This operator goes from right to left. To make this work correctly, we
  // always add the new layer to the end of the list. This also lets you use
  // parens, e.g. `model --> (layer1 --> layer2) --> layer3`.
  lhs.lastLayer.next = rhs
  return lhs
}

public func --> (lhs: Model, rhs: Layer) -> Model {
  // FUTURE: writing `model --> layer` adds the first layer to a dictionary,
  // using the layer's name as the key. This lets you add multiple inputs to
  // the model by writing `model --> layer` multiple times. When predicting,
  // you feed a dictionary of textures into predict().
  lhs.firstLayer = rhs
  return lhs
}

public typealias ParameterCallback = (String, Int, ParameterType) -> ParameterData?

enum ModelError: Error {
  case compileError(message: String)
}

class ModelCompiler {
  let device: MTLDevice
  let parameterCallback: ParameterCallback
  let inflightBuffers: Int

  var numLayers = 0
  var imageDescriptors: [DataShape: MPSImageDescriptor] = [:]
  var outputImages: [String: [MPSImage]] = [:]

  // Only used when the very first layer accepts a texture instead of an image.
  var firstLayerEatsTexture = false
  var firstComputeLayer = true
  var sourceTexture: MTLTexture?

  init(device: MTLDevice, inflightBuffers: Int, parameterCallback: @escaping ParameterCallback) {
    self.device = device
    self.inflightBuffers = inflightBuffers
    self.parameterCallback = parameterCallback
  }

  /**
    Goes through the given chain of layers and fills in the output shapes.
    Also registers MPSImageDescriptors for these shapes.
  */
  func calculateOutputShapes(for layer: Layer) throws {
    var ptr: Layer? = layer
    while ptr != nil, let node = ptr {
      numLayers += 1

      if node.name.isEmpty {
        node.name = "__\(node.typeName)_\(numLayers)__"
      }

      if !node.inputShape.isFullySpecified && !node.allowsIncompleteShape {
        throw ModelError.compileError(message: "input shape \(node.inputShape) for layer '\(node)' has unknown dimensions")
      }

      try node.calculateOutputShapesForChildren(compiler: self)

      let shape = node.outputShape(for: node.inputShape)
      node.outputShape = shape

      if shape.isFullySpecified {
        registerImageDescriptor(for: shape)
      }

      // The passthrough layer is just a trick that lets us optionally add
      // layers or not. We don't actually want them in the model.
      if node.next is PassthroughLayer {
        node.next = node.next!.next
      }

      ptr = node.next
      ptr?.inputShape = shape
    }
  }

  func registerImageDescriptor(for shape: DataShape) {
    if imageDescriptors[shape] == nil {
      imageDescriptors[shape] = shape.createImageDescriptor()
    }
  }

  /**
    Creates MPSCNN objects for the specified layer.
  */
  func createComputeForLayer(_ node: Layer) throws {
    // FUTURE: Sort the layers by largest weights to smallest, in order to
    // load the largest layers first. This makes it possible to load very
    // big models on devices with limited memory capacity, since the params
    // need to be copied into MPSCNN and therefore are in memory twice for
    // a short while.

    var weightParameters: ParameterData?
    if node.weightCount > 0 {
      weightParameters = parameterCallback(node.name, node.weightCount, .weights)
    }

    var biasParameters: ParameterData?
    if node.biasCount > 0 {
      biasParameters = parameterCallback(node.name, node.biasCount, .biases)
    }

    try node.createComputeForChildren(compiler: self)

    try node.createCompute(device: device,
                           weights: weightParameters,
                           biases: biasParameters)

    // Does the first layer take a MTLTexture or an MPSImage?
    if node.needsEncoding && firstComputeLayer {
      if node.wantsTextures { firstLayerEatsTexture = true }
      firstComputeLayer = false
    }

    // Make an MPSImage for any layer that asks for a real image instead of
    // a temporary one. We keep track of these in a dictionary.
    if !node.imageIsTemporary {
      addOutputImage(for: node)
    }
  }

  func addOutputImage(for node: Layer) {
    guard let imgDesc = imageDescriptors[node.outputShape] else {
      fatalError("Error: could not find image descriptor for shape \(node.outputShape)")
    }

    // Since the GPU can be working on several inputs at once, we need to
    // allocate multiple images.
    var array: [MPSImage] = []
    for _ in 0..<inflightBuffers {
      array.append(MPSImage(device: device, imageDescriptor: imgDesc))
    }
    outputImages[node.name] = array
  }

  /**
    Creates the GPU commands for a layer and its children (if any).
    
    - Parameters:
      - sourceImage: The MPSImage with the input. Is never nil, except for the
        very first layer if that layer wants a texture instead of an image.
      - destinationImage: The MPSImage that should contain the output. This is
        usually nil, in which case a new destination image will automatically
        be allocated for the layer. If not nil, the layer will write into the
        given image (useful for merging multiple layers into the same image).
  */
  func encode(commandBuffer: MTLCommandBuffer,
              layer node: Layer,
              sourceImage: MPSImage?,
              destinationImage: MPSImage?,
              inflightIndex: Int) -> MPSImage? {

    // Create a new destination image or reuse the given one.
    var destinationImage = destinationImage
    if node.needsDestinationImage {
      // If the node has a real MPSImage, use that. Otherwise make a temp one.
      if let images = outputImages[node.name] {
        destinationImage = images[inflightIndex]
      } else {
        guard let desc = imageDescriptors[node.outputShape] else {
          fatalError("Error: no image descriptor found for shape \(node.outputShape)")
        }
        destinationImage = MPSTemporaryImage(commandBuffer: commandBuffer,
                                             imageDescriptor: desc)
      }
    }

    node.encodeChildren(compiler: self,
                        commandBuffer: commandBuffer,
                        sourceImage: sourceImage,
                        destinationImage: destinationImage,
                        inflightIndex: inflightIndex)

    if node.needsEncoding {
      guard let destinationImage = destinationImage else {
        fatalError("Error: expected destination image")
      }

      if node.wantsTextures {
        let inputTexture: MTLTexture
        if let sourceImage = sourceImage {
          inputTexture = sourceImage.texture
        } else if let sourceTexture = sourceTexture {
          inputTexture = sourceTexture   // valid only for first layer
        } else {
          fatalError("Error: expected source texture")
        }

        node.encode(commandBuffer: commandBuffer,
                    sourceTexture: inputTexture,
                    destinationTexture: destinationImage.texture)
      } else {
        guard let sourceImage = sourceImage else {
          fatalError("Error: expected source image")
        }

        node.encode(commandBuffer: commandBuffer,
                    sourceImage: sourceImage,
                    destinationImage: destinationImage)
      }
    }

    return destinationImage
  }
}

/**
  The top-level object for the neural network.
  
  You first add layers to the model using `-->` and then call `compile()` to
  construct all the Metal objects.
*/
public class Model {
  var firstLayer: Layer?

  var compiled = false
  var compiler: ModelCompiler!
  var imageDescriptorList: [MPSImageDescriptor] = []
  var lastLayerName = ""

  public init() { }

  /**
    Creates all the MPSCNN objects for this graph.
    
    - Parameters:
      - inflightBuffers: How many tasks the CPU and GPU can do in parallel.
      - parameterCallback: Used for loading the parameters of the layers.
        The closure takes three arguments: name, expected parameter count, and
        whether to load the weights or bias values for the layer.
  */
  public func compile(device: MTLDevice,
                      inflightBuffers: Int,
                      parameterCallback: @escaping ParameterCallback) -> Bool {
    if compiled {
      print("Compile error: graph has already been compiled")
      return false
    }

    guard let layer = firstLayer else {
      print("Compile error: model needs at least one layer")
      return false
    }

    let startTime = CACurrentMediaTime()

    compiler = ModelCompiler(device: device,
                             inflightBuffers: inflightBuffers,
                             parameterCallback: parameterCallback)

    do {
      // First we calculate how large the input shapes and output shapes of all
      // the layers are (including nested layers). This also fills up the cache
      // with MPSImageDescriptors.
      try compiler.calculateOutputShapes(for: layer)

      // Create the compute objects for all the layers.
      var ptr: Layer? = layer
      while ptr != nil, let node = ptr {
        try compiler.createComputeForLayer(node)

        // Always make an MPSImage for the last layer.
        if node.next == nil {
          compiler.addOutputImage(for: node)
          lastLayerName = node.name
        }

        ptr = node.next
      }

      imageDescriptorList = Array(compiler.imageDescriptors.values)

      let elapsed = CACurrentMediaTime() - startTime
      print("Compiling took \(elapsed) seconds")

      compiled = true
      return true

    } catch ModelError.compileError(let message) {
      print("Compile error:", message)
      return false
    } catch {
      print("Unknown error: \(error)")
      return false
    }
  }

  /**
    Creates the GPU commands for a forward pass of the neural network.

    - FUTURE: Takes a dict with textures for multiple inputs. The inputs are
      identified by name.
              
    - FUTURE: We want encoding to be as fast as possible, so we could move all
      the encoding logic into compile(), which then outputs a list of commands.
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

    if !compiled {
      print("Error: graph has not been compiled yet")
      return
    }

    MPSTemporaryImage.prefetchStorage(with: commandBuffer,
                                      imageDescriptorList: imageDescriptorList)

    var sourceImage: MPSImage!
    if compiler.firstLayerEatsTexture {
      compiler.sourceTexture = texture
    } else {
      sourceImage = MPSImage(texture: texture, featureChannels: 3)
    }

    var ptr: Layer? = firstLayer
    while ptr != nil, let node = ptr {
      sourceImage = compiler.encode(commandBuffer: commandBuffer,
                                    layer: node,
                                    sourceImage: sourceImage,
                                    destinationImage: nil,
                                    inflightIndex: inflightIndex)
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
    s += "Layer                          Type       Output Shape     Parameters\n"
    s += "---------------------------------------------------------------------\n"

    var totalParams = 0
    var ptr: Layer? = firstLayer
    while ptr != nil, let node = ptr {
      let (text, params) = node.summary()
      s += text
      totalParams += params
      ptr = node.next
    }

    s += "---------------------------------------------------------------------\n"
    s += "Number of layers: \(compiler.numLayers)\n"
    s += "Total parameters: \(totalParams)"
    return s
  }

  /**
    Find a layer by name. You can use this before compiling to change the
    properties of a layer (only some properties can be changed, most notably
    `imageIsTemporary`).
    
    Currently you cannot find layers that are inside Groups. (But for those
    layers it doesn't make sense to change `imageIsTemporary` anyway, as they
    write in the Group's image.)
  */
  public subscript(name: String) -> Layer? {
    var ptr: Layer? = firstLayer
    while ptr != nil, let node = ptr {
      if node.name == name { return node }
      ptr = node.next
    }
    return nil
  }

  func images(forLayer name: String) -> [MPSImage] {
    return compiler.outputImages[name] ?? []
  }

  /** 
    Returns the output from the given layer. This layer must have its
    `imageIsTemporary` property set to false!
  */
  public func image(forLayer name: String, inflightIndex: Int) -> MPSImage {
    return images(forLayer: name)[inflightIndex]
  }

  /** Returns the output from the last layer in the model. */
  public func outputImage(inflightIndex: Int) -> MPSImage {
    return image(forLayer: lastLayerName, inflightIndex: inflightIndex)
  }
}
