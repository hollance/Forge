import Metal
import MetalPerformanceShaders
import Forge

class Kernel {
  let device: MTLDevice
  let pipeline: MTLComputePipelineState
  let name: String

  init(device: MTLDevice, functionName: String) {
    self.device = device
    self.name = functionName
    pipeline = makeFunction(device: device, name: functionName)
  }

  func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
    let encoder = commandBuffer.makeComputeCommandEncoder()
    encoder.pushDebugGroup(name)
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(sourceImage.texture, at: 0)
    encoder.setTexture(destinationImage.texture, at: 1)
    encoder.dispatch(pipeline: pipeline, rows: destinationImage.texture.height, columns: destinationImage.texture.width)
    encoder.popDebugGroup()
    encoder.endEncoding()
  }
}

class MakeGrayscale: Kernel {
  init(device: MTLDevice) {
    super.init(device: device, functionName: "makeGrayscale")
  }
}

extension Kernel: CustomKernel { }
