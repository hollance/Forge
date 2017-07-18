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
import MetalKit
import MetalPerformanceShaders

/**
  To use your neural network with Runner it must conform to this protocol.
  
  For optimal throughput we don't want the CPU to wait for the GPU, or vice
  versa. This means the GPU can be working on several inputs at once. Runner
  takes care of the synchronization for you.
  
  However, the NeuralNetwork must allocate multiple output images so that each 
  independent GPU pass gets its own MPSImage. (You need to do this for all 
  MPSImages stored by the neural network, but not for MPSTemporaryImages.)
*/
public protocol NeuralNetwork {
  associatedtype PredictionType

  /**
    Encodes the commands for the GPU.

    - Parameters:
      - texture: the MTLTexture with the image or video frame to process
      - inflightIndex: which output image to use for this GPU pass
  */
  func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int)

  /**
    Converts the output MPSImage into an array of predictions.
    This is called from a background thread.
  */
  func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<PredictionType>
}

/**
  This object is passed back to the UI thread after the neural network has
  made a new prediction.
*/
public struct NeuralNetworkResult<PredictionType> {
  public var predictions: [PredictionType] = []

  // For debugging purposes it can be useful to look at the output from
  // intermediate layers. To do so, make the layer write to a real MPSImage
  // object (not MPSTemporaryImage) and fill in the debugTexture property.
  // The UI thread can then display this texture as a UIImage.
  public var debugTexture: MTLTexture?
  public var debugScale: Float = 1       // for scaling down float images
  public var debugOffset: Float = 0      // for images with negative values

  // This is filled in by Runner to measure the latency between starting a
  // prediction and receiving the answer. (NOTE: Because we can start a new
  // prediction while the previous one is still being processed, the latency
  // actually becomes larger the more inflight buffers you're using. It is
  // therefore *not* a good indicator of throughput, i.e. frames per second.)
  public var latency: CFTimeInterval = 0

  public init() { }
}

/**
  Runner is a simple wrapper around the neural network that takes care of
  scheduling the GPU commands and so on. This leaves the NeuralNetwork object
  free to just do neural network stuff.
*/
public class Runner {
  public let device: MTLDevice
  public let commandQueue: MTLCommandQueue

  let inflightSemaphore: DispatchSemaphore
  let inflightBuffers: Int
  var inflightIndex = 0

  /**
    - Parameters:
      - inflightBuffers: How many tasks the CPU and GPU can do in parallel.
        Typical value is 3. Use 1 if you want the CPU to always wait until 
        the GPU is done (this is not recommended).
  */
  public init(commandQueue: MTLCommandQueue, inflightBuffers: Int) {
    self.device = commandQueue.device
    self.commandQueue = commandQueue
    self.inflightBuffers = inflightBuffers
    self.inflightSemaphore = DispatchSemaphore(value: inflightBuffers)
  }

  /**
    Encodes the commands for the GPU, commits the command buffer, and returns
    immediately. It does not wait for the GPU to finish.
    
    When the GPU finishes executing the buffer, the results are sent to the 
    completion handler, which will run on the specified queue.

    - Note: This method *can* block until the GPU is ready to receive commands
      again! You should call it from a background thread -- it's OK to use the
      VideoCapture queue for this.
  */
  public func predict<NeuralNetworkType: NeuralNetwork>(
                      network: NeuralNetworkType,
                      texture inputTexture: MTLTexture,
                      queue: DispatchQueue,
                      completion: @escaping (NeuralNetworkResult<NeuralNetworkType.PredictionType>) -> Void) {

    // Block until the next GPU buffer is available.
    inflightSemaphore.wait()

    let startTime = CACurrentMediaTime()

    //commandQueue.insertDebugCaptureBoundary()

    autoreleasepool {
      guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

      network.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)

      // The completion handler for the command buffer is called on some
      // background thread. This may be the same thread that encoded the
      // GPU commands (if not waiting on the semaphore), or another one.
      commandBuffer.addCompletedHandler { [inflightIndex] commandBuffer in

        var result = network.fetchResult(inflightIndex: inflightIndex)
        result.latency = CACurrentMediaTime() - startTime

        //print("GPU execution duration:", commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        //print("Elapsed time: \(endTime - startTime) sec")

        queue.async { completion(result) }

        // We're done, so wake up the encoder thread if it's waiting.
        self.inflightSemaphore.signal()
      }

      inflightIndex = (inflightIndex + 1) % inflightBuffers
      commandBuffer.commit()
    }
  }
}
