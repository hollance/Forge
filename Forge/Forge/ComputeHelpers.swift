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

import Foundation
import Metal
import MetalPerformanceShaders

var defaultMetalLibrary: MTLLibrary!
var forgeMetalLibrary: MTLLibrary!

func loadDefaultMetalLibrary(device: MTLDevice) -> MTLLibrary {
  if defaultMetalLibrary == nil {
    defaultMetalLibrary = device.newDefaultLibrary()
    if defaultMetalLibrary == nil {
      fatalError("Could not load default Metal library")
    }
  }
  return defaultMetalLibrary
}

func loadForgeMetalLibrary(device: MTLDevice) -> MTLLibrary {
  if forgeMetalLibrary == nil {
    guard let path = Bundle(for: Runner.self).path(forResource: "default",
                                                   ofType: "metallib") else {
      fatalError("Could not find Forge Metal library")
    }
    do {
      forgeMetalLibrary = try device.makeLibrary(filepath: path)
    } catch {
      fatalError("Could not load Forge Metal library")
    }
  }
  return forgeMetalLibrary
}

/**
  Creates a pipeline for a compute kernel.
*/
public func makeFunction(device: MTLDevice, name: String,
                         constantValues: MTLFunctionConstantValues? = nil,
                         useForgeLibrary: Bool = false) -> MTLComputePipelineState {

  let library = useForgeLibrary ? loadForgeMetalLibrary(device: device)
                                : loadDefaultMetalLibrary(device: device)

  return makeFunction(library: library, name: name, constantValues: constantValues)
}

/**
  Helper function that creates a pipeline for a compute kernel.
*/
public func makeFunction(library: MTLLibrary, name: String,
                         constantValues: MTLFunctionConstantValues? = nil) -> MTLComputePipelineState {
  do {
    if let constantValues = constantValues {
      let kernelFunction = try library.makeFunction(name: name, constantValues: constantValues)
      return try library.device.makeComputePipelineState(function: kernelFunction)
    } else {
      guard let kernelFunction = library.makeFunction(name: name) else {
        fatalError("Could not load compute function '\(name)'")
      }
      return try library.device.makeComputePipelineState(function: kernelFunction)
    }
  } catch {
    fatalError("Could not create compute pipeline for function '\(name)'")
  }
}

extension MTLComputeCommandEncoder {
  /**
    Sets the parameters of the command encoder with less boilerplate.
    
    Example:
    
        encoder.configure(parameters: [someBuffer, someTexture, UInt32(someValue)])
    
    MTLBuffer objects are passed to the shader with `setBuffer()`, MTLTexture
    objects with `setTexture()`, and anything else with `setBytes()`.

    - Parameters:
      - parameters: an array that may contain MTLBuffer or MTLTexture objects, 
        or types such as UInt32 or structs
  */
  public func configure(parameters: [Any]) {
    for i in 0..<parameters.count {
      var obj = parameters[i]
      if let buffer = obj as? MTLBuffer {
        setBuffer(buffer, offset: 0, at: i)
      } else if let texture = obj as? MTLTexture {
        setTexture(texture, at: i)
      } else {
        setBytes(&obj, length: MemoryLayout.size(ofValue: obj), at: i)
      }
    }
  }
}

extension MTLComputeCommandEncoder {
  /**
    Dispatches a compute kernel on a 1-dimensional grid.
    
    - Parameters:
      - count: the number of elements to process
  */
  public func dispatch(pipeline: MTLComputePipelineState, count: Int) {
    // Round off count to the nearest multiple of threadExecutionWidth.
    let width = pipeline.threadExecutionWidth
    let rounded = ((count + width - 1) / width) * width

    let blockSize = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
    let numBlocks = (count + blockSize - 1) / blockSize

    let threadGroupSize = MTLSizeMake(blockSize, 1, 1)
    let threadGroups = MTLSizeMake(numBlocks, 1, 1)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }

  /**
    Dispatches a compute kernel on a 2-dimensional grid.
    
    - Parameters:
      - rows: the first dimension
      - columns: the second dimension
  */
  public func dispatch(pipeline: MTLComputePipelineState, rows: Int, columns: Int) {
    let h = pipeline.threadExecutionWidth
    let w = pipeline.maxTotalThreadsPerThreadgroup / h

    let threadGroupSize = MTLSizeMake(w, h, 1)

    let threadGroups = MTLSizeMake(
      (rows    + threadGroupSize.width  - 1) / threadGroupSize.width,
      (columns + threadGroupSize.height - 1) / threadGroupSize.height, 1)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }

  /**
    Dispatches a compute kernel on an MPSImage's texture or texture array.
  */
  public func dispatch(pipeline: MTLComputePipelineState, image: MPSImage) {
    let numSlices = ((image.featureChannels + 3)/4) * image.numberOfImages

    let h = pipeline.threadExecutionWidth
    let w = pipeline.maxTotalThreadsPerThreadgroup / h
    let d = 1
    let threadGroupSize = MTLSizeMake(w, h, d)

    let threadGroups = MTLSizeMake(
      (image.width  + threadGroupSize.width  - 1) / threadGroupSize.width,
      (image.height + threadGroupSize.height - 1) / threadGroupSize.height,
      (numSlices    + threadGroupSize.depth  - 1) / threadGroupSize.depth)
    
    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }
}

/**
  Copies user-supplied weights into an MTLBuffer.

  Because we're working with textures, the number of channels must always be
  a multiple of 4. For example, the first row of a 3x3 kernel with 4 or fewer
  channels looks like this in memory: 0,1,2,3 | 0,1,2,3 | 0,1,2,3. And for 4-8 
  channels it looks like: 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7.

  But when featureChannels is not a multiple of 4, the weights supplied by the
  user may look like this: 0,1,2 | 0,1,2 | 0,1,2 and so on. In that case we
  need to insert zero bytes for the missing channels when copying the weights
  to the MTLBuffer.
*/
func weightsToBufferFloat16(weights: UnsafePointer<Float>,
                            buffer: MTLBuffer,
                            featureChannels: Int,
                            count: Int) {

  let slices = (featureChannels + 3) / 4
  let paddedChannels = slices * 4

  // The number of channels is a multiple of 4, so we can do a straight copy.
  if paddedChannels == featureChannels {
    let ptr = UnsafeMutablePointer(mutating: weights)
    float32to16(input: ptr, output: buffer.contents(), count: count)

  // Otherwise, copy "featureChannels" weights at a time and add padding bytes.
  } else {
    var srcPtr = UnsafeMutablePointer(mutating: weights)
    var dstPtr = buffer.contents().bindMemory(to: Float16.self, capacity: count)

    for _ in 0..<count/slices {
      float32to16(input: srcPtr, output: dstPtr, count: featureChannels)
      srcPtr += featureChannels
      dstPtr += paddedChannels
    }
  }
}
