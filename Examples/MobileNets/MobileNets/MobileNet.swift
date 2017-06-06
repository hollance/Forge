import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import Forge

/**
  The neural network from the paper "MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications" https://arxiv.org/abs/1704.04861v1
*/
class MobileNet: NeuralNetwork {
  typealias Prediction = (labelIndex: Int, probability: Float)

  let classes: Int
  let model: Model

  /**
    Creates a new MobileNet object.
    
    - Parameters:
      - widthMultiplier: Shrinks the number of channels. This is a value in the
        range (0, 1]. Default is 1, which starts the network with 32 channels.
        (This hyperparameter is called "alpha" in the paper.)
      - resolutionMultiplier: Shrink the spatial dimensions of the input image.
        This is a value between (0, 1]. Default is 1, which resizes to 224x224
        pixels. (The paper calls this hyperparameter "rho".)
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

    let input = Input()

    var x = input
        --> Resize(width: resolution, height: resolution)
        --> Custom(Preprocessing(device: device), channels: 3)
        --> Convolution(kernel: (3, 3), channels: channels, stride: (2, 2), activation: relu, name: "conv1")
        --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv2_1_dw")
        --> PointwiseConvolution(channels: channels*2, activation: relu, name: "conv2_1_sep")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv2_2_dw")
        --> PointwiseConvolution(channels: channels*4, activation: relu, name: "conv2_2_sep")
        --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv3_1_dw")
        --> PointwiseConvolution(channels: channels*4, activation: relu, name: "conv3_1_sep")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv3_2_dw")
        --> PointwiseConvolution(channels: channels*8, activation: relu, name: "conv3_2_sep")
        --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv4_1_dw")
        --> PointwiseConvolution(channels: channels*8, activation: relu, name: "conv4_1_sep")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv4_2_dw")
        --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv4_2_sep")

    if !shallow {
      x = x --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv5_1_dw")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv5_1_sep")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv5_2_dw")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv5_2_sep")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv5_3_dw")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv5_3_sep")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv5_4_dw")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv5_4_sep")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv5_5_dw")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv5_5_sep")
    }

    x = x --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv5_6_dw")
          --> PointwiseConvolution(channels: channels*32, activation: relu, name: "conv5_6_sep")
          --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv6_dw")
          --> PointwiseConvolution(channels: channels*32, activation: relu, name: "conv6_sep")
          --> GlobalAveragePooling()
          --> Dense(neurons: classes, activation: nil, name: "fc7")
          --> Softmax()

    model = Model(input: input, output: x)

    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 suffix: type == .weights ? "_w" : "_b",
                                                 ext: "bin")
    }

    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()
    assert(probabilities.count == (self.classes / 4) * 4)

    var result = NeuralNetworkResult<Prediction>()
    result.predictions = probabilities.top(k: 5).map { x -> Prediction in (x.0, x.1) }
    return result
  }
}
