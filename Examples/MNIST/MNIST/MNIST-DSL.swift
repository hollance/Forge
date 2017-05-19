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
class MNIST: NeuralNetwork {
  typealias Prediction = (label: String, probability: Float)

  let model: Model
  let resizeLayer: Resize
  let grayscale: Tensor

  init(device: MTLDevice, inflightBuffers: Int) {
    let relu = MPSCNNNeuronReLU(device: device, a: 0)

    let input = Input()

    // Keep a reference to this layer so we can crop the image on-the-fly.
    resizeLayer = Resize(width: 28, height: 28)

    grayscale = input
            --> resizeLayer
            --> Custom(Preprocessing(device: device), channels: 1)

    // Make the output of the Preprocessing kernel a real MPSImage so we can
    // show it on the screen (for debugging purposes).
    grayscale.imageIsTemporary = false

    let output = grayscale
             --> Convolution(kernel: (5, 5), channels: 20, activation: relu, name: "conv1")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Convolution(kernel: (5, 5), channels: 50, activation: relu, name: "conv2")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2))
             --> Dense(neurons: 320, activation: relu, name: "fc1")
             --> Dense(neurons: 10, name: "fc2")
             --> Softmax()

    model = Model(input: input, output: output)

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

  func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    // This is how you can dynamically crop the input texture before resizing
    // (this crops the input image to the center square).
    resizeLayer.setCropRect(x: 0, y: 60, width: 360, height: 360)

    model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)
  }

  func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    // Convert the MTLTexture from outputImage into something we can use
    // from Swift and then find the class with the highest probability.
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()

    // Note: there are only 10 classes but the number of channels in the 
    // output texture will always be a multiple of 4; in this case 12.
    assert(probabilities.count == 12)

    let (maxIndex, maxValue) = probabilities.argmax()

    var result = NeuralNetworkResult<Prediction>()
    result.predictions.append((label: "\(maxIndex)", probability: maxValue))

    // Enable this to see the output of the preprocessing shader.
    result.debugTexture = model.image(for: grayscale, inflightIndex: inflightIndex).texture
    result.debugScale = 1/255

    return result
  }
}
