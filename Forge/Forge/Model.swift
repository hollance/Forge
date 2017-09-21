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
  associativity: left
  higherThan: MultiplicationPrecedence
}

infix operator --> : ChainPrecedence

public func --> (lhs: Tensor, rhs: Layer) -> Tensor {
  return Tensor(input: lhs, layer: rhs)
}

public typealias ParameterCallback = (String, Int, ParameterType) -> ParameterData?

public enum ModelError: Error {
  case compileError(message: String)
}

/**
  The top-level object for the neural network.
  
  How to build and use the model:
  
  1. first create layers and tensors,
  2. instantiate a `Model` using the input and output tensor,
  3. call `compile()` to construct all the Metal objects,
  4. use `summary()` to verify that your model is correct,
  5. use `encode()` to perform inference.
*/
public class Model {
  let input: Tensor
  let output: Tensor

  var compiled = false
  var tensors: [Tensor] = []
  var imageDescriptors: [DataShape: MPSImageDescriptor] = [:]
  var imageDescriptorList: [MPSImageDescriptor] = []
  var outputImages: [Tensor: [MPSImage]] = [:]
  var numLayers = 0
  var totalParams = 0

  // Used during compiling.
  var device: MTLDevice!
  var parameterCallback: ParameterCallback!
  var inflightBuffers = 0

  // Only used when the very first layer accepts a texture instead of an image.
  var firstLayerEatsTexture = false
  var firstComputeLayer = true
  var sourceTexture: MTLTexture?

  public init(input: Tensor, output: Tensor) {
    self.input = input
    self.output = output
  }

  deinit {
    freeTensors(&tensors)
  }

  func freeTensors(_ tensors: inout [Tensor]) {
    for tensor in tensors {
      freeTensors(&tensor.next)
    }
    tensors = []
  }

  /**
    Creates all the MPSCNN objects for this graph.
    
    - Parameters:
      - inflightBuffers: How many tasks the CPU and GPU can do in parallel.
      - parameterCallback: Used for loading the parameters of the layers.
        The closure takes three arguments: name, expected parameter count, and
        whether to load the weights or bias values for the layer.
  */
  public func compile(device: MTLDevice,
                      inflightBuffers: Int = 3,
                      parameterCallback: @escaping ParameterCallback) -> Bool {
    if compiled {
      print("Compile error: graph has already been compiled")
      return false
    }

    let startTime = CACurrentMediaTime()

    self.device = device
    self.inflightBuffers = inflightBuffers
    self.parameterCallback = parameterCallback

    do {
      tensors = topologicalSort(from: input)

      try completeGraph()
      try createComputeForAllLayers()

      imageDescriptorList = Array(imageDescriptors.values)
      for imageDesc in imageDescriptorList {
        imageDesc.storageMode = .private
      }

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
    This gives us an array of tensors in an order that is guaranteed to be
    correct (but possibly is not the order the model was specified in).
  */
  func topologicalSort(from source: Tensor) -> [Tensor] {
    var stack = [Tensor]()
    var visited = Set<Tensor>()

    func depthFirstSearch(_ source: Tensor) {
      for neighbor in source.next {
        if !visited.contains(neighbor) {
          depthFirstSearch(neighbor)
        }
      }
      stack.append(source)
      visited.insert(source)
    }

    depthFirstSearch(source)
    return stack.reversed()
  }

  /**
    Makes sure the graph can actually be compiled and fills in any missing
    information. This also fills up the cache with MPSImageDescriptors.
  */
  func completeGraph() throws {
    for (i, tensor) in tensors.enumerated() {
      if let layer = tensor.layer {
        numLayers += 1

        // Assign a name to any layers that don't have one.
        if layer.name.isEmpty {
          layer.name = "__\(layer.typeName)_\(i+1)__"
        }

        // If the layer expects a fully-specified shape but the previous layers
        // didn't fill in the width/height/depth, then we cannot continue.
        if let input = tensor.input, !input.shape.isFullySpecified
                                  && !layer.allowsIncompleteShape {
          throw ModelError.compileError(message: "input shape \(input.shape) for layer '\(layer)' has unknown dimensions")
        }
      }

      if tensor.shape.isFullySpecified {
        registerImageDescriptor(for: tensor.shape)
      }
    }
  }

  func registerImageDescriptor(for shape: DataShape) {
    if imageDescriptors[shape] == nil {
      imageDescriptors[shape] = shape.createImageDescriptor()
    }
  }

  /**
    Creates compute kernels for the layers in the graph. Also allocates any
    non-temporary MPSImages for tensors that want them.
  */
  func createComputeForAllLayers() throws {
    for tensor in tensors {
      if let layer = tensor.layer, let input = tensor.input {
        // Only make the compute once for each layer (important for layers
        // that get reused).
        if !layer.createdCompute {
          try createCompute(for: layer, input: input, output: tensor)
          layer.createdCompute = true
        }
      }

      // Make an MPSImage for any tensor that asks for a real image instead
      // of a temporary one. We keep track of these in a dictionary.
      if !tensor.imageIsTemporary {
        addOutputImage(for: tensor)
      }
    }

    // Always make an MPSImage for the last tensor.
    if let output = tensors.last {
      addOutputImage(for: output)
    }
  }

  /**
    Creates Metal objects for the specified layer.
  */
  func createCompute(for layer: Layer, input: Tensor, output: Tensor) throws {
    // FUTURE: Sort the layers by largest weights to smallest, in order to
    // load the largest layers first. This makes it possible to load very
    // big models on devices with limited memory capacity, since the params
    // need to be copied into MPSCNN and therefore are in memory twice for
    // a short while.

    //print("createCompute:", input, "-->", output)

    var weightParameters: ParameterData?
    let weightCount = layer.weightCount(inputShape: input.shape, outputShape: output.shape)
    if weightCount > 0 {
      totalParams += weightCount
      weightParameters = parameterCallback(layer.name, weightCount, .weights)
    }

    var biasParameters: ParameterData?
    let biasCount = layer.biasCount(inputShape: input.shape, outputShape: output.shape)
    if biasCount > 0 {
      totalParams += biasCount
      biasParameters = parameterCallback(layer.name, biasCount, .biases)
    }

    try layer.createCompute(device: device,
                            inputShape: input.shape,
                            outputShape: output.shape,
                            weights: weightParameters,
                            biases: biasParameters)

    layer.paramCount = weightCount + biasCount

    // Does the first layer take a MTLTexture or an MPSImage?
    if firstComputeLayer {
      if layer.wantsTextures { firstLayerEatsTexture = true }
      firstComputeLayer = false
    }
  }

  func addOutputImage(for tensor: Tensor) {
    guard let imgDesc = imageDescriptors[tensor.shape] else {
      fatalError("Error: could not find image descriptor for shape \(tensor.shape)")
    }

    // Since the GPU can be working on several inputs at once, we need to
    // allocate multiple images.
    var array: [MPSImage] = []
    for _ in 0..<inflightBuffers {
      array.append(MPSImage(device: device, imageDescriptor: imgDesc))
    }
    outputImages[tensor] = array
  }

  /**
    Returns a summary of all the layers and tensors in the model, including
    their types, shapes, and number of parameters. Useful for making sure your
    model is correct.
  */
  public func summary() -> String {
    guard compiled else { return "(Model is not compiled.)" }

    var s = ""
    s += "Layer/Tensor                   Type       Output Shape     Parameters\n"
    s += "---------------------------------------------------------------------\n"

    for tensor in tensors {
      s += tensor.summary() + "\n"
    }

    s += "---------------------------------------------------------------------\n"
    s += "Number of layers: \(numLayers) (tensors: \(tensors.count))\n"
    s += "Total parameters: \(totalParams)"
    return s
  }

  /**
    Encodes the GPU commands for a forward pass of the neural network.
  */
  public func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int) {
    //let startTime = CACurrentMediaTime()

    if !compiled {
      print("Error: graph has not been compiled yet")
      return
    }

    MPSTemporaryImage.prefetchStorage(with: commandBuffer,
                                      imageDescriptorList: imageDescriptorList)

    var sourceImage: MPSImage?
    if firstLayerEatsTexture {
      sourceTexture = texture
    } else {
      sourceImage = MPSImage(texture: texture, featureChannels: 3)
    }

    tensors[0].image = sourceImage

    for tensor in tensors {
      encode(tensor: tensor, commandBuffer: commandBuffer, inflightIndex: inflightIndex)
    }

    //let elapsed = CACurrentMediaTime() - startTime
    //print("Encoding took \(elapsed) seconds")
  }

  func encode(tensor: Tensor, commandBuffer: MTLCommandBuffer, inflightIndex: Int) {
    // If a tensor does not have a layer (true for Input and Concatenate), then 
    // pass through the source image unchanged.
    guard let layer = tensor.layer else { return }

    // If the tensor has a real MPSImage, use that. Otherwise make a temp one.
    func createImage(for tensor: Tensor) -> MPSImage {
      if let images = outputImages[tensor] {
        return images[inflightIndex]
      } else {
        guard let desc = imageDescriptors[tensor.shape] else {
          fatalError("Error: no image descriptor found for shape \(tensor.shape)")
        }
        let image = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: desc)
        image.readCount = tensor.readCount
        return image
      }
    }

    // If this tensor does not use its own image, then grab the one from the
    // destination tensor. Otherwise, make a new image.
    if let destTensor = tensor.destinationTensor {
      if let image = destTensor.image {
        tensor.image = image
      } else {
        destTensor.readCount = destTensor.next.count
        destTensor.image = createImage(for: destTensor)
        tensor.image = destTensor.image
      }
    } else {
      tensor.readCount = tensor.next.count
      tensor.image = createImage(for: tensor)
    }

    guard let inputTensor = tensor.input else {
      fatalError("Error: missing source tensor")
    }

    if layer.wantsTextures {
      let inputTexture: MTLTexture
      if let sourceImage = inputTensor.image {
        inputTexture = sourceImage.texture
      } else if let sourceTexture = sourceTexture {
        inputTexture = sourceTexture   // valid only for first layer
      } else {
        fatalError("Error: layer '\(layer.name)' expected source texture")
      }

      layer.encode(commandBuffer: commandBuffer,
                   sourceTensor: inputTensor,
                   sourceTexture: inputTexture,
                   destinationTensor: tensor)

      if let image = inputTensor.image as? MPSTemporaryImage {
        image.readCount -= 1
      }
    } else {
      layer.encode(commandBuffer: commandBuffer,
                   sourceTensor: inputTensor,
                   destinationTensor: tensor)
    }

    // At this point we've used the image from the sourceTensor, and should
    // decrement its reference count. When it hits 0, we nil out its `image`
    // property so that a new MPSTemporaryImage will be allocated on the next
    // pass through the network.
    inputTensor.readCount -= 1
    if inputTensor.readCount <= 0 {
      inputTensor.image = nil
    }
  }

  func images(for tensor: Tensor) -> [MPSImage] {
    return outputImages[tensor] ?? []
  }

  /** 
    Returns the output from the given tensor. This tensor must have its
    `imageIsTemporary` property set to false!
  */
  public func image(for tensor: Tensor, inflightIndex: Int) -> MPSImage {
    return images(for: tensor)[inflightIndex]
  }

  /** Returns the output from the last tensor in the model. */
  public func outputImage(inflightIndex: Int) -> MPSImage {
    return image(for: output, inflightIndex: inflightIndex)
  }
}
