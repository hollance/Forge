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
  var outputImage: [MPSImage] = []
  let classes: Int

  let relu: MPSCNNNeuronReLU
  let lanczos: MPSImageLanczosScale
  var layers: [Any] = []
  let softmax: MPSCNNSoftMax

  var imgDesc: [MPSImageDescriptor] = []
  var layerImgDesc: [Int] = []
  let outputImgDesc: MPSImageDescriptor

  let channelFormat = MPSImageFeatureChannelFormat.float16

  /**
    Creates a new MobileNet object.
    
    - Parameters:
      - alpha: Width multiplier, a value between (0, 1]. Default is 1.
      - rho: Resolution multiplier, a value between (0, 1]. Default is 1.
      - shallow: Whether to exclude the group of 5 conv layers in the middle.
      - classes: The number of classes in the softmax.
  */
  required public init(device: MTLDevice,
                       alpha: Float = 1,
                       rho: Float = 1,
                       shallow: Bool = false,
                       classes: Int = 1000,
                       inflightBuffers: Int) {

    self.classes = classes

    outputImgDesc = MPSImageDescriptor(channelFormat: channelFormat,
                                       width: 1,
                                       height: 1,
                                       featureChannels: 1000)
    for _ in 0..<inflightBuffers {
      outputImage.append(MPSImage(device: device, imageDescriptor: outputImgDesc))
    }

    relu = MPSCNNNeuronReLU(device: device, a: 0)
    lanczos = MPSImageLanczosScale(device: device)
    softmax = MPSCNNSoftMax(device: device)

    weightsLoader = { name, count in ParameterLoaderRandom(count: count) }
    biasLoader = { name, count in ParameterLoaderRandom(count: count) }

    addLayers(device: device, alpha: alpha, rho: rho, shallow: shallow, classes: classes)
  }

  func addLayers(device: MTLDevice, alpha: Float, rho: Float, shallow: Bool, classes: Int) {
    var currentWidth = Int(224 * rho)
    var currentHeight = Int(224 * rho)
    var currentChannels = Int(32 * alpha)

    func addImageDescriptor(_ desc: MPSImageDescriptor) {
      print("\t\(desc.width) x \(desc.height) x \(desc.featureChannels)")
      imgDesc.append(desc)
    }

    func useImageDescriptor() {
      layerImgDesc.append(imgDesc.count - 1)
    }

    func addDepthwiseStride2(name: String) {
      layers.append(depthwiseConvolution(device: device,
                                         kernel: (3, 3),
                                         channels: currentChannels,
                                         name: name,
                                         stride: (2, 2)))
      currentWidth /= 2
      currentHeight /= 2
      addImageDescriptor(MPSImageDescriptor(channelFormat: channelFormat,
                                            width: currentWidth,
                                            height: currentHeight,
                                            featureChannels: currentChannels))
      useImageDescriptor()

      layers.append(relu)
      useImageDescriptor()
    }

    func addDepthwiseStride1(name: String) {
      layers.append(depthwiseConvolution(device: device,
                                         kernel: (3, 3),
                                         channels: currentChannels,
                                         name: name))
      useImageDescriptor()

      layers.append(relu)
      useImageDescriptor()
    }

    func addPointwise(name: String) {
      layers.append(pointwiseConvolution(device: device,
                                         inChannels: currentChannels,
                                         outChannels: currentChannels,
                                         filter: relu, name: name))
      useImageDescriptor()
    }

    func addPointwise2x(name: String) {
      layers.append(pointwiseConvolution(device: device,
                                         inChannels: currentChannels,
                                         outChannels: currentChannels * 2,
                                         filter: relu, name: name))
      currentChannels *= 2
      addImageDescriptor(MPSImageDescriptor(channelFormat: channelFormat,
                                            width: currentWidth,
                                            height: currentHeight,
                                            featureChannels: currentChannels))
      useImageDescriptor()
    }

    addImageDescriptor(MPSImageDescriptor(channelFormat: channelFormat,
                                          width: currentWidth,
                                          height: currentHeight,
                                          featureChannels: 3))

    layers.append(convolution(device: device,
                              kernel: (3, 3),
                              inChannels: 3,
                              outChannels: currentChannels,
                              filter: relu,
                              name: "conv1_s2",
                              stride: (2, 2)))
    currentWidth /= 2
    currentHeight /= 2

    addImageDescriptor(MPSImageDescriptor(channelFormat: channelFormat,
                                          width: currentWidth,
                                          height: currentHeight,
                                          featureChannels: currentChannels))
    useImageDescriptor()

    addDepthwiseStride1(name: "conv2_dw_s1")
    addPointwise2x(name: "conv3_pw_s1")

    addDepthwiseStride2(name: "conv4_dw_s2")
    addPointwise2x(name: "conv5_pw_s1")

    addDepthwiseStride1(name: "conv6_dw_s1")
    addPointwise(name: "conv7_pw_s1")

    addDepthwiseStride2(name: "conv8_dw_s2")
    addPointwise2x(name: "conv9_pw_s1")

    addDepthwiseStride1(name: "conv10_dw_s1")
    addPointwise(name: "conv11_pw_s1")

    addDepthwiseStride2(name: "conv12_dw_s2")
    addPointwise2x(name: "conv13_pw_s1")

    if !shallow {
      for i in 0..<5 {
        addDepthwiseStride1(name: "conv\(14+i*2)_dw_s1")
        addPointwise(name: "conv\(15+i*2)_pw_s1")
      }
    }

    addDepthwiseStride2(name: "conv24_dw_s2")
    addPointwise2x(name: "conv25_pw_s1")

    // NOTE: the paper says this is stride 2, but that does not
    // correspond to the given input size for the next layer.
    addDepthwiseStride1(name: "conv26_dw_s1")
    addPointwise(name: "conv27_pw_s1")

    let fcChannels = Int(1024 * alpha)

    layers.append(averagePooling(device: device, kernel: (7, 7), stride: (7, 7)))
    addImageDescriptor(MPSImageDescriptor(channelFormat: channelFormat,
                                          width: 1,
                                          height: 1,
                                          featureChannels: fcChannels))
    useImageDescriptor()

    layers.append(dense(device: device, fanIn: fcChannels, fanOut: classes, filter: nil, name: "fc28"))
    addImageDescriptor(outputImgDesc)
    useImageDescriptor()
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: imgDesc)

    let scaledImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: imgDesc[0])
    lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: scaledImage.texture)

    var inputImage: MPSImage = scaledImage
    for (i, layer) in layers.enumerated() {
      let img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: imgDesc[layerImgDesc[i]])
      if let layer = layer as? MPSCNNKernel {
        layer.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: img)
      } else if let layer = layer as? DepthwiseConvolutionKernel {
        layer.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: img)
      }
      inputImage = img
    }

    softmax.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage[inflightIndex])
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult {
    // Convert the MTLTexture from outputImage into something we can use
    // from Swift and then find the class with the highest probability.
    let probabilities = outputImage[inflightIndex].toFloatArray()
    assert(probabilities.count == self.classes)
    let (maxIndex, maxValue) = probabilities.argmax()

    var result = NeuralNetworkResult()
    result.predictions.append((label: "\(maxIndex)", probability: maxValue))
    return result
  }
}
