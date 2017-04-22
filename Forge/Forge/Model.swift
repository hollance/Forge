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

public typealias ParameterCallback = (String, Int, ParameterType) -> ParameterData?

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
    
    - Parameters:
      - inflightBuffers: How many tasks the CPU and GPU can do in parallel.
      - parameterCallback: Used for loading the parameters of the layers.
        The closure takes three arguments: name, expected parameter count, and
        whether to load the weights or bias values for the layer.
  */
  public func compile(device: MTLDevice,
                      inflightBuffers: Int,
                      parameterCallback: ParameterCallback) -> Bool {
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
        print("Compile error: input shape \(node.inputShape) for layer '\(node)' has unknown dimensions")
        return false
      }

      let shape = node.outputShape(for: node.inputShape)
      node.outputShape = shape

      if shape.isFullySpecified {
        if imageDescriptors[shape] == nil {
          imageDescriptors[shape] = shape.createImageDescriptor()
        }

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

        if let compute = node.createCompute(device: device,
                                            weights: weightParameters,
                                            biases: biasParameters) {
          node.compute = compute

          // Does the first layer take a MTLTexture or an MPSImage?
          if firstComputeLayer {
            if case .mps = compute { firstLayerEatsTexture = true }
            firstComputeLayer = false
          }
        } else {
          print("Compile error: could not create compute object for layer '\(node)'")
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
