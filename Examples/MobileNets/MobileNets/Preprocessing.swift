import Metal
import Forge

public class Preprocessing: SimpleKernel {
  public init(device: MTLDevice) {
    super.init(device: device, functionName: "preprocess")
  }
}
