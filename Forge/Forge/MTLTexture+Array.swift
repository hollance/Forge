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

import Metal

extension MTLTexture {
  /**
    Convenience function that copies the texture's pixel data to a Swift array.

    The type of `initial` determines the type of the output array. In the
    following example, the type of bytes is `[UInt8]`.

        let bytes = texture.toArray(width: 100, height: 100, channels: 4, initial: UInt8(0))

    - Parameters:
      - channels: The number of color components per pixel: must be 1, 2, or 4.
      - initial: This parameter is necessary because we need to give the array 
        an initial value. Unfortunately, we can't do `[T](repeating: T(0), ...)` 
        since `T` could be anything and may not have an init that takes a literal 
        value.
  */
  public func toArray<T>(width: Int, height: Int, channels: Int, initial: T) -> [T] {
    assert(channels != 3 && channels <= 4, "channels must be 1, 2, or 4")

    var bytes = [T](repeating: initial, count: width * height * channels)
    let region = MTLRegionMake2D(0, 0, width, height)
    getBytes(&bytes, bytesPerRow: width * channels * MemoryLayout<T>.stride,
             from: region, mipmapLevel: 0)
    return bytes
  }
}

extension MTLDevice {
  /**
    Convenience function that makes a new texture from a Swift array.
    
    - Parameters:
      - channels: The number of color components per pixel: must be 1, 2 or 4.
   */
  public func makeTexture<T>(array: [T],
                             width: Int,
                             height: Int,
                             channels: Int,
                             pixelFormat: MTLPixelFormat) -> MTLTexture {

    assert(channels != 3 && channels <= 4, "channels must be 1, 2, or 4")

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: pixelFormat, width: width, height: height, mipmapped: false)

    let texture = makeTexture(descriptor: textureDescriptor)

    let region = MTLRegionMake2D(0, 0, width, height)
    texture.replace(region: region, mipmapLevel: 0, withBytes: array,
                    bytesPerRow: width * MemoryLayout<T>.stride * channels)
    return texture
  }
}

/*
  NOTE: The method device.makeTexture(array:...) copies a Swift array into an
  MTLTexture that you can then turn into an MPSImage. However, this does not
  let you create an image with > 4 channels.
  
  It's possible to create the following extension on MPSImage:

      MPSImage(device:, array: [T], featureChannels:, numberOfImages:)
    
  This first creates the MPSImage object, then loops through the texture array
  and copies the array's bytes into it. It would be the opposite of the method
  MPSImage.toFloatArray().
  
  However, it would have the same restrictions: for 3 channels you'd have to
  add padding into the Swift array yourself. And for > 4 channels you have to
  organize the array by slices. So I don't know how useful it is in practice.
  Most of the time inputs to the network will be 1 or 4 channels.
*/
