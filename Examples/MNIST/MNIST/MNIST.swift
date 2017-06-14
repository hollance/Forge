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
  Implements a neural network like LeNet-5, trained on MNIST.
  
  It has the following layers:
  
  - Input: 28x28 image (grayscale)
  - Conv 5x5 kernel, 1 input channel, 20 output channels, ReLU
  - Max Pooling 2x2 kernel, stride 2
  - Conv 5x5 kernel, 20 input channels, 50 output channels, ReLU
  - Max Pooling 2x2 kernel, stride 2
  - Fully-connected 320 units, ReLU
  - Fully-connected 10 units, softmax
*/
class MNIST: NeuralNetwork {
  typealias Prediction = (label: String, probability: Float)

  var outputImage: [MPSImage] = []

  let lanczos: MPSImageLanczosScale
  let makeGrayscale: Preprocessing
  let relu: MPSCNNNeuronReLU
  let softmax: MPSCNNSoftMax

  let conv1: MPSCNNConvolution
  let pool1: MPSCNNPoolingMax
  let conv2: MPSCNNConvolution
  let pool2: MPSCNNPoolingMax
  let fc1: MPSCNNFullyConnected
  let fc2: MPSCNNFullyConnected

  let scaledImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 3)
  let grayImgDesc   = MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 1)
  let conv1ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 20)
  let pool1ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 14, height: 14, featureChannels: 20)
  let conv2ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 14, height: 14, featureChannels: 50)
  let pool2ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width:  7, height:  7, featureChannels: 50)
  let fc1ImgDesc    = MPSImageDescriptor(channelFormat: .float16, width:  1, height:  1, featureChannels: 320)
  let outputImgDesc = MPSImageDescriptor(channelFormat: .float16, width:  1, height:  1, featureChannels: 10)

  let grayImg: MPSImage

  let temporaryImageDescriptors: [MPSImageDescriptor]

  public init(device: MTLDevice, inflightBuffers: Int) {
    // Since the GPU can be working on several inputs at once, this needs to
    // allocate multiple output images.
    for _ in 0..<inflightBuffers {
      outputImage.append(MPSImage(device: device, imageDescriptor: outputImgDesc))
    }

    lanczos = MPSImageLanczosScale(device: device)
    makeGrayscale = Preprocessing(device: device)
    relu = MPSCNNNeuronReLU(device: device, a: 0)
    softmax = MPSCNNSoftMax(device: device)

    weightsLoader = { name, count in ParameterLoaderBundle(name: name, count: count, suffix: "_W", ext: "bin") }
    biasLoader = { name, count in ParameterLoaderBundle(name: name, count: count, suffix: "_b", ext: "bin") }

    conv1 = convolution(device: device, kernel: (5, 5), inChannels: 1, outChannels: 20, activation: relu, name: "conv1")
    pool1 = maxPooling(device: device, kernel: (2, 2), stride: (2, 2))
    conv2 = convolution(device: device, kernel: (5, 5), inChannels: 20, outChannels: 50, activation: relu, name: "conv2")
    pool2 = maxPooling(device: device, kernel: (2, 2), stride: (2, 2))
    fc1 = dense(device: device, shape: (7, 7), inChannels: 50, fanOut: 320, activation: relu, name: "fc1")
    fc2 = dense(device: device, fanIn: 320, fanOut: 10, activation: nil, name: "fc2")

    // I want to show the output of the preprocessing shader on the screen for
    // debugging, so store its results in a real MSPImage, not a temporary one.
    grayImg = MPSImage(device: device, imageDescriptor: grayImgDesc)

    temporaryImageDescriptors = [
      scaledImgDesc, /*grayImgDesc,*/ conv1ImgDesc, pool1ImgDesc,
      conv2ImgDesc, pool2ImgDesc, fc1ImgDesc, outputImgDesc
    ]

    // Temporary images must have private storage mode.
    for imgDesc in temporaryImageDescriptors {
      imgDesc.storageMode = .private
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    // This lets us squeeze some extra speed out of Metal.
    MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: temporaryImageDescriptors)

    // Resize the input image to 28x28 pixels.
    let scaledImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: scaledImgDesc)
    lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: scaledImg.texture)

    // Convert from RGB (3 channels) to grayscale (1 channel) since this
    // network is trained on 1-channel images. This shader also scales the
    // numbers up by a factor 255.
    //let grayImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: grayImgDesc)
    makeGrayscale.encode(commandBuffer: commandBuffer, sourceImage: scaledImg, destinationImage: grayImg)

    let conv1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1ImgDesc)
    conv1.applyPadding(type: .same, sourceImage: grayImg, destinationImage: conv1Img)
    conv1.encode(commandBuffer: commandBuffer, sourceImage: grayImg, destinationImage: conv1Img)

    let pool1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool1ImgDesc)
    pool1.encode(commandBuffer: commandBuffer, sourceImage: conv1Img, destinationImage: pool1Img)

    let conv2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2ImgDesc)
    conv2.applyPadding(type: .same, sourceImage: pool1Img, destinationImage: conv2Img)
    conv2.encode(commandBuffer: commandBuffer, sourceImage: pool1Img, destinationImage: conv2Img)

    let pool2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool2ImgDesc)
    pool2.encode(commandBuffer: commandBuffer, sourceImage: conv2Img, destinationImage: pool2Img)

    let fc1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1ImgDesc)
    fc1.encode(commandBuffer: commandBuffer, sourceImage: pool2Img, destinationImage: fc1Img)

    let fc2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: outputImgDesc)
    fc2.encode(commandBuffer: commandBuffer, sourceImage: fc1Img, destinationImage: fc2Img)

    // Finally, apply the softmax function to the output of the last layer.
    // The output image is not an MPSTemporaryImage but a regular MSPImage.
    softmax.encode(commandBuffer: commandBuffer, sourceImage: fc2Img, destinationImage: outputImage[inflightIndex])
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    // Convert the MTLTexture from outputImage into something we can use
    // from Swift and then find the class with the highest probability.
    // Note: there are only 10 classes but the number of channels in the 
    // output texture will always be a multiple of 4; in this case 12.
    let probabilities = outputImage[inflightIndex].toFloatArray()
    assert(probabilities.count == 12)
    let (maxIndex, maxValue) = probabilities.argmax()

    var result = NeuralNetworkResult<Prediction>()
    result.predictions.append((label: "\(maxIndex)", probability: maxValue))

    // Enable this to see the output of the preprocessing shader.
    result.debugTexture = grayImg.texture
    result.debugScale = 1/255
    return result
  }
}
