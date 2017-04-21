import Metal
import MetalPerformanceShaders

/**
  Converts a 3 or 4 channel input image into a 1 channel image.
*/
public class RGB2Gray: SimpleKernel {
  public init(device: MTLDevice) {
    super.init(device: device, functionName: "rgb2Gray", useForgeLibrary: true)
  }
}
