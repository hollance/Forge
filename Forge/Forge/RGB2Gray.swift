import Metal
import MetalPerformanceShaders

public class RGB2Gray: SimpleKernel {
  public init(device: MTLDevice) {
    super.init(device: device, functionName: "rgb2Gray", useForgeLibrary: true)
  }
}
