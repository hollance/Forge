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

import MetalPerformanceShaders

extension MPSImage {
  /**
    We receive the predicted output as an MPSImage. We need to convert this
    to an array of Floats that we can use from Swift.

    Because Metal is a graphics API, MPSImage stores the data in MTLTexture 
    objects. Each pixel from the texture stores 4 channels: R contains the 
    first channel, G is the second channel, B is the third, A is the fourth. 

    In addition, these individual R,G,B,A pixel components can be stored as 
    `.float16`, in which case we also have to convert the data type.

    ---WARNING---

    If there are > 4 channels in the MPSImage, then the channels are organized
    in the output as follows:

        [ 1,2,3,4,1,2,3,4,...,1,2,3,4,5,6,7,8,5,6,7,8,...,5,6,7,8 ]
      
    and not as you'd expect:

        [ 1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8,...,1,2,3,4,5,6,7,8 ]

    First are channels 1 - 4 for the entire image, followed by channels 5 - 8
    for the entire image, and so on. That happens because we copy the data out
    of the texture by slice, and we can't interleave slices.

    If the number of channels is not a multiple of 4, then the output will
    have padding bytes in it:

        [ 1,2,3,4,1,2,3,4,...,1,2,3,4,5,6,-,-,5,6,-,-,...,5,6,-,- ]

    The size of the array is therefore always a multiple of 4! So if you have
    a classifier for 10 classes, the output vector is 12 elements and the last
    two elements are zero.

    The only case where you get the kind of array you'd actually expect is
    when the number of channels is 1, 2, or 4 (i.e. there is only one slice):
    
        [ 1,1,1,...,1 ] or [ 1,2,1,2,1,2,...,1,2 ] or [ 1,2,3,4,...,1,2,3,4 ]
  */
  @nonobjc public func toFloatArray() -> [Float] {
    switch pixelFormat {
      case .r16Float, .rg16Float, .rgba16Float: return fromFloat16()
      case .r32Float, .rg32Float, .rgba32Float: return fromFloat32()
      default: fatalError("Pixel format \(pixelFormat) not supported")
    }
  }

  private func fromFloat16() -> [Float] {
    var outputFloat16 = convert(initial: Float16(0))
    return float16to32(&outputFloat16, count: outputFloat16.count)
  }

  private func fromFloat32() -> [Float] {
    return convert(initial: Float(0))
  }

  private func convert<T>(initial: T) -> [T] {
    let numSlices = (featureChannels + 3)/4

    // If the number of channels is not a multiple of 4, we may need to add 
    // padding. For 1 and 2 channels we don't need padding.
    let channelsPlusPadding = (featureChannels < 3) ? featureChannels : numSlices * 4

    // Find how many elements we need to copy over from each pixel in a slice.
    // For 1 channel it's just 1 element (R); for 2 channels it is 2 elements 
    // (R+G), and for any other number of channels it is 4 elements (RGBA).
    let numComponents = (featureChannels < 3) ? featureChannels : 4

    // Allocate the memory for the array. If batching is used, we need to copy
    // numSlices slices for each image in the batch.
    let count = width * height * channelsPlusPadding * numberOfImages
    var output = [T](repeating: initial, count: count)

    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: width, height: height, depth: 1))

    for i in 0..<numSlices*numberOfImages {
      texture.getBytes(&(output[width * height * numComponents * i]),
                       bytesPerRow: width * numComponents * MemoryLayout<T>.stride,
                       bytesPerImage: 0,
                       from: region,
                       mipmapLevel: 0,
                       slice: i)
    }
    return output
  }
}

extension MPSImage {
  /**
    Creates an MPSImage from an array of Floats.
    
    The data must be arranged as follows: for each image in the batch, first 
    must be channels 0-3 for all pixels, followed by channels 4-7, followed by
    channels 8-11, etc. This is so we can copy the data into the MTLTextures
    slice-by-slice.

    NOTE: If the number of channels is not a multiple of 4, you need to add
    padding bytes. So if you want to make an image with only 3 channels, you
    still need to provide 4 channels of data.
  */
  public convenience init(device: MTLDevice,
                          numberOfImages: Int = 1,
                          width: Int,
                          height: Int,
                          featureChannels: Int,
                          array: UnsafeMutablePointer<Float>,
                          count: Int) {
    let imageDesc = MPSImageDescriptor(channelFormat: .float16,
                                       width: width,
                                       height: height,
                                       featureChannels: featureChannels,
                                       numberOfImages: numberOfImages,
                                       usage: [.shaderRead, .shaderWrite])

    self.init(device: device, imageDescriptor: imageDesc)

    let float16Input = float32to16(array, count: count)
    float16Input.withUnsafeBytes { buf in
      // We copy 4 channels worth of data at a time
      let bytesPerRow = width * 4 * MemoryLayout<Float16>.stride
      let slices = ((featureChannels + 3) / 4) * numberOfImages
      var ptr = buf.baseAddress!
      for s in 0..<slices {
        texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0,
                        slice: s,
                        withBytes: ptr,
                        bytesPerRow: bytesPerRow,
                        bytesPerImage: 0)
        ptr += height * bytesPerRow
      }
    }
  }
}

/**
  Creates an MPSImage from a file containing binary 32-floats. Useful for
  debugging.
  
  The pixel data must be arranged in slices of 4 channels.

  In Python, given an image of shape `(batchSize, height, width, channels)` 
  you can convert it as follows to the expected format:

      X = np.zeros((batchSize, height, width, channels)).astype(np.float32)
      slices = []
      for j in range(X.shape[0]):
          for i in range(X.shape[3] // 4):
              slice = X[j, :, :, i*4:(i+1)*4].reshape(1, X.shape[1], X.shape[2], 4)
              slices.append(slice)
      X_metal = np.concatenate(slices, axis=0)
      X_metal.tofile("X.bin")
*/
public func loadBinaryFloatImage(device: MTLDevice,
                                 url: URL,
                                 numberOfImages: Int = 1,
                                 width: Int,
                                 height: Int,
                                 featureChannels: Int) -> MPSImage {
  let data = try! Data(contentsOf: url)
  let inputSize = numberOfImages * height * width * featureChannels
  var float32Input = [Float](repeating: 0, count: inputSize)
  _ = float32Input.withUnsafeMutableBufferPointer { ptr in
    data.copyBytes(to: ptr)
  }

  return MPSImage(device: device,
                  numberOfImages: numberOfImages,
                  width: width,
                  height: height,
                  featureChannels: featureChannels,
                  array: &float32Input,
                  count: inputSize)
}

/**
  Creates an image with random pixel values between 0 and 1. Useful for testing
  and debugging.
*/
public func randomImage(device: MTLDevice,
                        numberOfImages: Int = 1,
                        width: Int,
                        height: Int,
                        featureChannels: Int,
                        seed: Int) -> MPSImage {

  let slices = (featureChannels + 3) / 4
  let inputSize = numberOfImages * height * width * slices * 4
  var float32Input = [Float](repeating: 0, count: inputSize)
  Random.uniformRandom(&float32Input, count: inputSize, scale: 1.0, seed: seed)

  return MPSImage(device: device,
                  numberOfImages: numberOfImages,
                  width: width,
                  height: height,
                  featureChannels: featureChannels,
                  array: &float32Input,
                  count: inputSize)
}
