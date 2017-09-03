import Foundation
import MetalPerformanceShaders
import Forge

class DepthwiseConvolutionTests {
  func runTest(imageSize: (Int, Int), channels: Int, stride: Int,
               filter: MPSCNNNeuron?, versusMPSDepthWise: Bool) {
    print("  channels: \(channels), stride: \(stride)")

    let kernelWidth = 3
    let kernelHeight = 3
    let imageWidth = imageSize.0
    let imageHeight = imageSize.0

    let depthwiseCount = channels * kernelWidth * kernelHeight
    var depthwiseWeights = [Float](repeating: 0, count: depthwiseCount)
    Random.uniformRandom(&depthwiseWeights, count: depthwiseCount, scale: 0.1, seed: time(nil))

    var biases = [Float](repeating: 0, count: channels)
    Random.uniformRandom(&biases, count: channels, scale: 0.3, seed: time(nil))

    let inputImage = randomImage(device: device, width: imageWidth,
                                 height: imageHeight, featureChannels: channels,
                                 seed: time(nil))

    let paddingX = (kernelWidth - 1)/2
    let paddingY = (kernelHeight - 1)/2
    let outputWidth = (imageWidth + 2*paddingX - kernelWidth) / stride + 1
    let outputHeight = (imageHeight + 2*paddingY - kernelHeight) / stride + 1
    //print(imageWidth, imageHeight, outputWidth, outputHeight)

    let outputImageDesc = MPSImageDescriptor(channelFormat: .float16,
                                       width: outputWidth,
                                       height: outputHeight,
                                       featureChannels: channels)

    let outputImage1 = MPSImage(device: device, imageDescriptor: outputImageDesc)
    let outputImage2 = MPSImage(device: device, imageDescriptor: outputImageDesc)

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

    let desc: MPSCNNConvolutionDescriptor
    var convWeights: [Float]

    if versusMPSDepthWise {
      if #available(iOS 11.0, *) {
        desc = MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: kernelWidth,
                                                    kernelHeight: kernelHeight,
                                                    inputFeatureChannels: channels,
                                                    outputFeatureChannels: channels,
                                                    neuronFilter: filter)
        convWeights = depthwiseWeights
      } else {
        fatalError("MPSCNNDepthWiseConvolutionDescriptor is only available on iOS 11 or later")
      }
    } else {
      desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                         kernelHeight: kernelHeight,
                                         inputFeatureChannels: channels,
                                         outputFeatureChannels: channels,
                                         neuronFilter: filter)

      // Transpose from [channels][kH][kW] to [kH][kW][channels].
      var transposedWeights = [Float](repeating: 0, count: depthwiseCount)
      for c in 0..<channels {
        for h in 0..<kernelHeight {
          for w in 0..<kernelWidth {
            transposedWeights[h*channels*kernelWidth + w*channels + c] = depthwiseWeights[c*kernelWidth*kernelHeight + h*kernelHeight + w]
          }
        }
      }

      // Running a depthwise convolution is like a regular convolution that
      // has a lot of its weights set to 0.
      //
      // The transposed weights for the depthwise convolution are stored as:
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
      convWeights = [Float](repeating: 0, count: convCount)
      let weightsPerSlice = kernelWidth * kernelHeight
      for c in 0..<channels {
        for w in 0..<weightsPerSlice {
          convWeights[weightsPerSlice*channels*c + w*channels + c] = transposedWeights[w*channels + c]
        }
      }
    }

    desc.strideInPixelsX = stride
    desc.strideInPixelsY = stride

    let conv = MPSCNNConvolution(device: device,
                                 convolutionDescriptor: desc,
                                 kernelWeights: convWeights,
                                 biasTerms: biases,
                                 flags: .none)
    conv.edgeMode = .zero

    let commandBuffer = commandQueue.makeCommandBuffer()!

    conv.applyPadding(type: .same, sourceImage: inputImage, destinationImage: outputImage2)
    depthwiseConv.offset = conv.offset
    print(conv.offset)

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

  func testCorrectness(versusMPSDepthWise: Bool) {
    print("\(self) versus MPSDepthWise: \(versusMPSDepthWise)")

    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let sigmoid = MPSCNNNeuronSigmoid(device: device)

    for i in [480, 97] {
      for c in [61, 13, 8, 4, 3, 2, 1] {
        for s in [1, 2, 3] {
          for f in [ relu, sigmoid, nil ] {
            runTest(imageSize: (i, i), channels: c, stride: s, filter: f, versusMPSDepthWise: versusMPSDepthWise)
          }
        }
      }
    }
  }

  /*
    Tests whether we can create a depthwise convolution by setting the number
    of channels of a regular convolution equal to the number of groups.
  */
  func testGroups() {
    print(#function)

    let kernelWidth = 3
    let kernelHeight = 3
    let imageWidth = 128
    let imageHeight = 128
    let channels = 64

    let count = kernelWidth * kernelHeight * channels
    var weights = [Float](repeating: 0, count: count)
    Random.uniformRandom(&weights, count: count, scale: 0.1, seed: time(nil))

    var biases = [Float](repeating: 0, count: channels)
    Random.uniformRandom(&biases, count: channels, scale: 0.3, seed: time(nil))

    let inputImage = randomImage(device: device, width: imageWidth,
                                 height: imageHeight, featureChannels: channels,
                                 seed: time(nil))

    let outputImageDesc = MPSImageDescriptor(channelFormat: .float16,
                                       width: imageWidth,
                                       height: imageHeight,
                                       featureChannels: channels)

    let outputImage1 = MPSImage(device: device, imageDescriptor: outputImageDesc)
    let outputImage2 = MPSImage(device: device, imageDescriptor: outputImageDesc)

    let depthwiseConv = DepthwiseConvolutionKernel(device: device,
                                                   kernelWidth: kernelWidth,
                                                   kernelHeight: kernelHeight,
                                                   featureChannels: channels,
                                                   strideInPixelsX: 1,
                                                   strideInPixelsY: 1,
                                                   channelMultiplier: 1,
                                                   neuronFilter: nil,
                                                   kernelWeights: weights,
                                                   biasTerms: biases)

    let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                           kernelHeight: kernelHeight,
                                           inputFeatureChannels: channels,
                                           outputFeatureChannels: channels,
                                           neuronFilter: nil)
    desc.strideInPixelsX = 1
    desc.strideInPixelsY = 1
    desc.groups = channels

    let conv = MPSCNNConvolution(device: device,
                                 convolutionDescriptor: desc,
                                 kernelWeights: weights,
                                 biasTerms: biases,
                                 flags: .none)
    conv.edgeMode = .zero

    let commandBuffer = commandQueue.makeCommandBuffer()!

    conv.applyPadding(type: .same, sourceImage: inputImage, destinationImage: outputImage2)
    depthwiseConv.offset = conv.offset
    print(conv.offset)

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
}
