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

var forgeMetalLibrary: MTLLibrary!

func loadForgeMetalLibrary(device: MTLDevice) -> MTLLibrary {
  if forgeMetalLibrary == nil {
    let bundle = Bundle(for: Runner.self)
    if let path = bundle.path(forResource: "default", ofType: "metallib") {
      do {
        forgeMetalLibrary = try device.makeLibrary(filepath: path)
      } catch {
        fatalError("Could not load Forge Metal library")
      }
    } else {
      fatalError("Could not find Forge Metal library")
    }
  }
  return forgeMetalLibrary
}

/**
  Creates a pipeline for a compute kernel using Forge's Metal library.
*/
func makeForgeFunction(device: MTLDevice, name: String) -> MTLComputePipelineState {
  return makeFunction(library: loadForgeMetalLibrary(device: device), name: name)
}

/**
  Creates a pipeline for a compute kernel using the default Metal library.
*/
public func makeFunction(device: MTLDevice, name: String) -> MTLComputePipelineState {
  guard let library = device.newDefaultLibrary() else {
    fatalError("Could not load default Metal library")
  }
  return makeFunction(library: library, name: name)
}

/**
  Helper function that creates a pipeline for a compute kernel.
*/
public func makeFunction(library: MTLLibrary, name: String) -> MTLComputePipelineState {
  do {
    guard let kernelFunction = library.makeFunction(name: name) else {
      fatalError("Could not load compute function '\(name)'")
    }
    return try library.device.makeComputePipelineState(function: kernelFunction)
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

    let h, w, d: Int
    if numSlices == 1 {
      h = pipeline.threadExecutionWidth
      w = pipeline.maxTotalThreadsPerThreadgroup / h
      d = 1
    } else {
      // TODO: Figure out the best way to divide up the work. Does it make
      // sense to work on 2 or more slices at once? Need to measure this!
      h = 16; w = 16; d = 2
    }

    let threadGroupSize = MTLSizeMake(w, h, d)
    let threadGroups = MTLSizeMake(
      (image.width  + threadGroupSize.width  - 1) / threadGroupSize.width,
      (image.height + threadGroupSize.height - 1) / threadGroupSize.height,
      (numSlices    + threadGroupSize.depth  - 1) / threadGroupSize.depth)
    
    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }
}
