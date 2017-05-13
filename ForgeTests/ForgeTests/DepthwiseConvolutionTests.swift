import Foundation
import MetalPerformanceShaders
import Forge

class DepthwiseConvolutionTests {
  func runTest(channels: Int, stride: Int, filter: MPSCNNNeuron?) {
    print("  channels: \(channels), stride: \(stride)")

    let kernelWidth = 3
    let kernelHeight = 3
    let imageWidth = 480
    let imageHeight = 360

    let depthwiseCount = kernelWidth * kernelHeight * channels
    var depthwiseWeights = [Float](repeating: 0, count: depthwiseCount)
    Random.uniformRandom(&depthwiseWeights, count: depthwiseCount, scale: 0.1, seed: time(nil))

    var biases = [Float](repeating: 0, count: channels)
    Random.uniformRandom(&biases, count: channels, scale: 0.3, seed: time(nil))

    let inputImage = randomImage(device: device, width: imageWidth,
                                 height: imageHeight, channels: channels,
                                 seed: time(nil))

    let imageDesc = MPSImageDescriptor(channelFormat: .float16,
                                       width: imageWidth,
                                       height: imageHeight,
                                       featureChannels: channels)

    let outputImage1 = MPSImage(device: device, imageDescriptor: imageDesc)
    let outputImage2 = MPSImage(device: device, imageDescriptor: imageDesc)

    let depthwiseConv = DepthwiseConvolutionKernel(device: device,
                                                   kernelWidth: kernelWidth,
                                                   kernelHeight: kernelHeight,
                                                   featureChannels: channels,
                                                   strideInPixelsX: stride,
                                                   strideInPixelsY: stride,
                                                   channelMultiplier: 1,
                                                   neuronFilter: filter,
                                                   kernelWeights: depthwiseWeights,
                                                   biasTerms: biases)

    let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                           kernelHeight: kernelHeight,
                                           inputFeatureChannels: channels,
                                           outputFeatureChannels: channels,
                                           neuronFilter: filter)
    desc.strideInPixelsX = stride
    desc.strideInPixelsY = stride

    // Running a depthwise convolution is like a regular convolution that
    // has a lot of its weights set to 0.
    //
    // The weights for the 3x3 depthwise convolution are stored as:
    //   1234 | 1234 | 1234
    //   1234 | 1234 | 1234
    //   1234 | 1234 | 1234
    // 
    // For the regular convolution we have to rearrange this as:
    //   1--- | 1--- | 1---
    //   1--- | 1--- | 1---
    //   1--- | 1--- | 1---
    //   -2-- | -2-- | -2--
    //   -2-- | -2-- | -2--
    //   -2-- | -2-- | -2--
    //   --3- | --3- | --3-
    // 
    // and so on... where the - represents a 0 value.

    let convCount = channels * kernelWidth * kernelHeight * channels
    var convWeights = [Float](repeating: 0, count: convCount)
    let weightsPerSlice = kernelWidth*kernelHeight
    for c in 0..<channels {
      for w in 0..<weightsPerSlice {
        convWeights[weightsPerSlice*channels*c + w*channels + c] = depthwiseWeights[w*channels + c]
      }
    }

    let conv = MPSCNNConvolution(device: device,
                                 convolutionDescriptor: desc,
                                 kernelWeights: convWeights,
                                 biasTerms: biases,
                                 flags: .none)
    conv.edgeMode = .zero

    let commandBuffer = commandQueue.makeCommandBuffer()

    depthwiseConv.encode(commandBuffer: commandBuffer,
                         sourceImage: inputImage,
                         destinationImage: outputImage1)

    conv.encode(commandBuffer: commandBuffer,
                sourceImage: inputImage,
                destinationImage: outputImage2)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let output1 = outputImage1.toFloatArray()
    let output2 = outputImage2.toFloatArray()

    assertEqual(output1, output2, tolerance: 1e-3)
  }

  func testCorrectness() {
    print("\(self)")

    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let sigmoid = MPSCNNNeuronSigmoid(device: device)

    for c in [61, 13, 8, 4, 3, 2, 1] {
      for s in [1, 2, 3] {
        for f in [ relu, sigmoid, nil ] {
          runTest(channels: c, stride: s, filter: f)
        }
      }
    }
  }
}
