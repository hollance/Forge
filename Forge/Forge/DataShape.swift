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

/**
  Describes the dimensions of the data as it flows through the neural net.

  Because Input can accept a texture of unknown size, we use -1 to indicate
  that a dimension is not known yet. (Don't want to use optionals for this,
  since most of the time the dimensions *will* be known and unwrapping just
  makes the code uglier.)
*/
public struct DataShape {
  public let width: Int
  public let height: Int
  public let channels: Int

  public init(width: Int = -1, height: Int = -1, channels: Int = -1) {
    self.width = width
    self.height = height
    self.channels = channels
  }

  var isFullySpecified: Bool {
    return width != -1 && height != -1 && channels != -1
  }

  func createImageDescriptor() -> MPSImageDescriptor {
    assert(isFullySpecified)
    return MPSImageDescriptor(channelFormat: .float16, width: width,
                              height: height, featureChannels: channels)
  }
}

extension DataShape: CustomDebugStringConvertible {
  public var debugDescription: String {
    var dims: [String] = []
    if width    != -1 { dims.append("\(width)")    } else { dims.append("?") }
    if height   != -1 { dims.append("\(height)")   } else { dims.append("?") }
    if channels != -1 { dims.append("\(channels)") } else { dims.append("?") }
    return "(" + dims.joined(separator: ", ") + ")"
  }
}

extension DataShape: Hashable {
  // Needs to be hashable because we'll create a cache of MPSImageDescriptor
  // objects. The DataShape is the key they're stored under.
  public var hashValue: Int {
    return width + height*1000 + channels*1000*1000
  }
}

public func == (lhs: DataShape, rhs: DataShape) -> Bool {
  return lhs.width    == rhs.width
      && lhs.height   == rhs.height
      && lhs.channels == rhs.channels
}
