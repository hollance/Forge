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

import UIKit
import Metal

extension UIImage {
  /**
    Converts an MTLTexture into a UIImage. This is useful for debugging.

    - TODO: This was not necessarily designed for speed. For more speed,
      look into using the vImage functions from Accelerate.framework or
      maybe CIImage.

    - Note: For `.float16` textures the pixels are expected to be in the range
      0...1; if you're using a different range (e.g. 0...255) then you have 
      to specify a `scale` factor and possibly an `offset`. Alternatively, you
      can use an `MPSNeuronLinear` to scale the pixels down first.
  */
  @nonobjc public class func image(texture: MTLTexture,
                                   scale: Float = 1,
                                   offset: Float = 0) -> UIImage {
    switch texture.pixelFormat {
    case .rgba16Float:
      return image(textureRGBA16Float: texture, scale: scale, offset: offset)
    case .r16Float:
      return image(textureR16Float: texture, scale: scale, offset: offset)
    case .rgba8Unorm:
      return image(textureRGBA8Unorm: texture)
    case .bgra8Unorm:
      return image(textureBGRA8Unorm: texture)
    case .r8Unorm:
      return image(textureR8Unorm: texture)
    default:
      fatalError("Unsupported pixel format \(texture.pixelFormat.rawValue)")
    }
  }

  @nonobjc class func image(textureRGBA16Float texture: MTLTexture,
                            scale: Float = 1,
                            offset: Float = 0) -> UIImage {

    // The texture must be `.float16` format. This means every RGBA pixel is
    // a 16-bit float, so one pixel takes up 64 bits.
    assert(texture.pixelFormat == .rgba16Float)

    let w = texture.width
    let h = texture.height

    // First get the bytes from the texture.
    var outputFloat16 = texture.toFloat16Array(width: w, height: h, featureChannels: 4)

    // Convert 16-bit floats to 32-bit floats.
    let outputFloat32 = float16to32(&outputFloat16, count: w * h * 4)

    // Convert the floats to bytes. The floats can go outside the range 0...1,
    // so we need to clamp the values when we turn them back into bytes.
    var outputRGBA = [UInt8](repeating: 0, count: w * h * 4)
    for i in 0..<outputFloat32.count {
      let value = outputFloat32[i] * scale + offset
      outputRGBA[i] = UInt8(max(min(255, value * 255), 0))
    }

    // Finally, turn the byte array into a UIImage.
    return UIImage.fromByteArray(&outputRGBA, width: w, height: h)
  }

  @nonobjc class func image(textureR16Float texture: MTLTexture,
                            scale: Float = 1,
                            offset: Float = 0) -> UIImage {

    assert(texture.pixelFormat == .r16Float)

    let w = texture.width
    let h = texture.height

    var outputFloat16 = texture.toFloat16Array(width: w, height: h, featureChannels: 1)
    let outputFloat32 = float16to32(&outputFloat16, count: w * h)

    var outputRGBA = [UInt8](repeating: 0, count: w * h * 4)
    for i in 0..<outputFloat32.count {
      let value = outputFloat32[i] * scale + offset
      let color = UInt8(max(min(255, value * 255), 0))
      outputRGBA[i*4 + 0] = color
      outputRGBA[i*4 + 1] = color
      outputRGBA[i*4 + 2] = color
      outputRGBA[i*4 + 3] = 255
    }

    return UIImage.fromByteArray(&outputRGBA, width: w, height: h)
  }

  @nonobjc class func image(textureRGBA8Unorm texture: MTLTexture) -> UIImage {
    assert(texture.pixelFormat == .rgba8Unorm)

    let w = texture.width
    let h = texture.height
    var bytes = texture.toUInt8Array(width: w, height: h, featureChannels: 4)
    return UIImage.fromByteArray(&bytes, width: w, height: h)
  }

  @nonobjc class func image(textureBGRA8Unorm texture: MTLTexture) -> UIImage {
    assert(texture.pixelFormat == .bgra8Unorm)

    let w = texture.width
    let h = texture.height
    var bytes = texture.toUInt8Array(width: w, height: h, featureChannels: 4)

    for i in 0..<bytes.count/4 {
      bytes.swapAt(i*4 + 0, i*4 + 2)
    }

    return UIImage.fromByteArray(&bytes, width: w, height: h)
  }

  @nonobjc class func image(textureR8Unorm texture: MTLTexture) -> UIImage {
    assert(texture.pixelFormat == .r8Unorm)

    let w = texture.width
    let h = texture.height
    var bytes = texture.toUInt8Array(width: w, height: h, featureChannels: 1)

    var rgbaBytes = [UInt8](repeating: 0, count: w * h * 4)
    for i in 0..<bytes.count {
      rgbaBytes[i*4 + 0] = bytes[i]
      rgbaBytes[i*4 + 1] = bytes[i]
      rgbaBytes[i*4 + 2] = bytes[i]
      rgbaBytes[i*4 + 3] = 255
    }

    return UIImage.fromByteArray(&rgbaBytes, width: w, height: h)
  }
}
