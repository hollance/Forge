import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import Forge

/**
  The neural network from the paper "MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications" https://arxiv.org/abs/1704.04861v1

  **NOTE:** This is currently using random parameters; the network *hasn't been 
  trained on anything yet*, so the predictions don't make any sense at all! 
  I just wanted to see how fast/slow this network architecture is on iPhone.
*/
public class MobileNet: NeuralNetwork {
  let classes: Int
  let model: Model

  /**
    Creates a new MobileNet object.
    
    - Parameters:
      - widthMultiplier: Shrinks the number of channels. This is a value in the
        range (0, 1]. Default is 1, which starts the network with 32 channels.
        (This hyperparmeter is called "alpha" in the paper.)
      - resolutionMultiplier: Shrink the spatial dimensions of the input image.
        This is a value between (0, 1]. Default is 1, which resizes to 224x224
        pixels. (The paper calls this hyperparmeter "rho".)
      - shallow: Whether to exclude the group of 5 conv layers in the middle.
      - classes: The number of classes in the softmax.
  */
  public init(device: MTLDevice,
              widthMultiplier: Float = 1,
              resolutionMultiplier: Float = 1,
              shallow: Bool = false,
              classes: Int = 1000,
              inflightBuffers: Int) {

    self.classes = classes

    let relu = MPSCNNNeuronReLU(device: device, a: 0)

    let channels = Int(32 * widthMultiplier)
    let resolution = Int(224 * resolutionMultiplier)

    let shallowLayers: Layer
    if shallow {
      shallowLayers = PassthroughLayer()
    } else {
      shallowLayers =
            DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*16, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*16, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*16, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*16, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*16, filter: relu, name: "TODO")
    }

    model = Model()
        --> Resize(width: resolution, height: resolution)
        --> Convolution(kernel: (3, 3), channels: channels, stride: (2, 2), filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*2, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*4, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*4, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*8, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*8, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*16, filter: relu, name: "TODO")
        --> shallowLayers
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*32, filter: relu, name: "TODO")
        --> DepthwiseConvolution(kernel: (3, 3), useReLU: true, name: "TODO")
        --> PointwiseConvolution(channels: channels*32, filter: relu, name: "TODO")
        --> GlobalAveragePooling()
        --> Dense(neurons: classes, filter: relu, name: "TODO")
        --> Softmax()

    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderRandom(count: count)
    }

    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult {
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()
    assert(probabilities.count == (self.classes / 4) * 4)
    let (maxIndex, maxValue) = probabilities.argmax()

    var result = NeuralNetworkResult()
    result.predictions.append((label: "\(maxIndex)", probability: maxValue))
    return result
  }
}
