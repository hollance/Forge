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
import Forge

/*
  Implements the LeNet-5 MNIST neural network using the DSL. This code is
  a lot shorter and easier to read!
*/
public class MNIST: NeuralNetwork {
  let model: Model

  public init(device: MTLDevice, inflightBuffers: Int) {
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let makeGrayscale = Preprocessing(device: device)

    model = Model()
            --> Input()        // this is optional thanks to Resize
            --> Resize(width: 28, height: 28, name: "Resizing")
            --> Custom(makeGrayscale, channels: 1, name: "Preprocessing")
            --> Convolution(kernel: (5, 5), channels: 20, filter: relu, name: "conv1")
            --> MaxPooling(kernel: (2, 2), stride: (2, 2))
            --> Convolution(kernel: (5, 5), channels: 50, filter: relu, name: "conv2")
            --> MaxPooling(kernel: (2, 2), stride: (2, 2))
            --> Dense(neurons: 320, filter: relu, name: "fc1")
            --> Dense(neurons: 10, name: "fc2")
            --> Softmax()

    model["Preprocessing"]!.imageIsTemporary = false

    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 suffix: type == .weights ? "_W" : "_b",
                                                 ext: "bin")
    }

    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    // This is how you can dynamically crop the input texture before resizing
    // (this crops the input image to the center square):
    if let resizing = model["Resizing"] as? Resize {
      resizing.setCropRegion(x: 0, y: 60, width: 360, height: 360)
    }

    model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult {
    // Convert the MTLTexture from outputImage into something we can use
    // from Swift and then find the class with the highest probability.
    // Note: there are only 10 classes but the number of channels in the 
    // output texture will always be a multiple of 4; in this case 12.
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()
    assert(probabilities.count == 12)
    let (maxIndex, maxValue) = probabilities.argmax()

    var result = NeuralNetworkResult()
    result.predictions.append((label: "\(maxIndex)", probability: maxValue))

    // Enable this to see the output of the preprocessing shader.
    result.debugTexture = model.image(forLayer: "Preprocessing", inflightIndex: inflightIndex).texture
    result.debugScale = 1/255
    return result
  }
}
