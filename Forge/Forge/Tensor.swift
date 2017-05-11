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
import MetalPerformanceShaders

/**
  A tensor has a shape (width, height, depth). For each tensor there is an 
  `MPS(Temporary)Image` that holds its data.

  The model is really a DAG of tensors: Each tensor knows what layer it came 
  from and what the input tensor to that layer was. Each tensor also knows
  what other tensors depend on it.
*/
public class Tensor {
  /**
    Whether this tensor writes its results into an MPSTemporaryImage. Normally
    this is true for all tensor except the last. You can override this if you
    want to keep track of the tensor's MPSImage for processing afterwards.
  */
  public var imageIsTemporary = true

  // The layer that takes the data from `input`, transforms it, and stores it
  // inside this tensor. Is nil for Input and Concatenate tensors.
  var layer: Layer?

  // Which tensor this one depends on. Together with `next`, this describes
  // the model's graph.
  var input: Tensor?

  // The tensors that depend on this one, i.e. which tensors we are the `input` 
  // for. The number of tensors in this array is also the readCount for this
  // tensor's MPSTemporaryImage.
  var next: [Tensor] = []

  // The shape of the tensor is determined by its input tensor and the layer.
  var shape = DataShape()

  // For debugging and printing the model summary.
  var typeName = "Tensor"

  // Used to set offset and clipRect for reading from another tensor's image.
  //var sourceChannelOffset = 0

  // Used to set destinationFeatureChannelOffset for merging the output from
  // multiple layers into one image.
  var destinationChannelOffset = 0

  // If this is set, the layer will write into the MPSImage for the destination
  // tensor. If nil, a new (temporary) image is allocated for the tensor and we
  // write into that. Usually this will be nil.
  var destinationTensor: Tensor?

  // The image that the layer for this tensor will write into. Since the layers
  // may not be processed in their original order (depending on the topological
  // sort), this is how we keep track of which MPSImage to use where. Note that
  // image may point to the destinationTensor's image.
  internal(set) public var image: MPSImage?

  // Reference count. It is used to set the readCount of the MPSTemporyImage
  // for this tensor, but also tells us when to set the `image` property to nil
  // when we're done using it (so that we don't hang on to images for longer
  // than necessary).
  var readCount = 1

  fileprivate init() { }

  /**
    Connects the `input` tensor to the `layer` and creates the output tensor
    that holds the results of the layer's computations.
    
    The shorthand way of writing this is:

        let output = input --> layer

    which just does:

        let output = Tensor(input: input, layer: layer)
  */
  public init(input: Tensor, layer: Layer) {
    self.input = input
    self.layer = layer

    input.next.append(self)
    shape = layer.outputShape(for: input.shape)
  }

  func summary() -> String {
    let layerName = layer?.name ?? "**\(typeName)**"
    let layerType = layer?.typeName ?? "Tensor"
    let paramCount = layer?.paramCount ?? 0

    let n = layerName.padding(toLength: 30, withPad: " ", startingAt: 0)
    let t = layerType.padding(toLength: 10, withPad: " ", startingAt: 0)
    let o = shape.debugDescription.padding(toLength: 16, withPad: " ", startingAt: 0)
    let p = "\(paramCount)".padding(toLength: 10, withPad: " ", startingAt: 0)

    let s = String(format: "%@ %@ %@ %@", n, t, o, p)
    //      + "\(destinationChannelOffset)"
    return s
  }
}

extension Tensor: Hashable {
  // Needs to be hashable because for tensors whose imageIsTemporary flag is
  // false, we use a dictionary to find the corresponding MPSImage objects.
  // Since each tensor is a unique entity, we use the tensor's address as the
  // hash value (this is similar to how NSObjects are hashed).
  public var hashValue: Int {
    return unsafeBitCast(self, to: Int.self)
  }
}

public func == (lhs: Tensor, rhs: Tensor) -> Bool {
  return lhs === rhs
}

extension Tensor: CustomDebugStringConvertible {
  public var debugDescription: String {
    return "Tensor, shape \(shape), layer " + (layer?.name ?? typeName)
  }
}

/**
  A placeholder for input. Your model always starts with an Input tensor.
  
  You can leave the shape of this tensor completely or partially unspecified.
  However, if you do specify a size, this is used to force the input texture 
  to be in a specific shape.
  
  If your first layer is `Resize`, which takes a texture of arbitrary size and
  scales it to a fixed size, then you can specify `Input()` without a shape.
  
  However, if your first layer is something like a `Convolution`, then you need
  `Input` to specify the size of the texture that goes into the conv layer. 
  (Without it, we won't know how large the `Convolution` layer's output will be
  and as a result we can't allocate an MPSTemporaryImage for it.)
*/
public func Input(width: Int? = nil, height: Int? = nil, channels: Int? = nil) -> Tensor {
  let tensor = Tensor()
  tensor.typeName = "Input"
  tensor.shape = DataShape(width: width ?? -1,
                           height: height ?? -1,
                           channels: channels ?? -1)
  return tensor
}

/**
  Depth-concatenates several tensors into one large tensor.
*/
public func Concatenate(_ tensors: [Tensor]) -> Tensor {
  let merged = Tensor()

  var maxWidth = 0
  var maxHeight = 0
  var channels = 0

  for input in tensors {
    // Tell the other tensor that it should write into our image and not
    // an image of its own.
    input.destinationChannelOffset = channels
    input.destinationTensor = merged

    // Figure out how large to make the merged tensor's destination image.
    maxWidth = max(maxWidth, input.shape.width)
    maxHeight = max(maxHeight, input.shape.height)
    channels += input.shape.channels

    // Connect each tensor to the merged tensor, or the topological sort
    // will fail and the graph will be incomplete.
    input.next.append(merged)
  }

  merged.shape = DataShape(width: maxWidth, height: maxHeight, channels: channels)
  merged.typeName = "Concat"

  // Note: We don't fill in the `input` property because we potentially have
  // multiple inputs, not just one. This is no problem because Concatenate is
  // skipped during encoding (as it has no layer).

  return merged
}
