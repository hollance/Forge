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

/*
  Encapsulates access to the weights/biases that are stored in a binary file.
  
  We only need to read from the parameters file while the neural network is
  being created. The weights are copied into the network (as 16-bit floats),
  so once the network is set up we no longer need to keep Parameters in memory.
*/
class Parameters {
  private var fileSize: Int
  private var fd: CInt!
  private var hdr: UnsafeMutableRawPointer!
  private(set) public var pointer: UnsafeMutablePointer<Float>!

  init?(path: String, fileSize: Int) {
    self.fileSize = fileSize

    fd = open(path, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
    if fd == -1 {
      print("Error: failed to open \"\(path)\", error = \(errno)")
      return nil
    }

    hdr = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
    if hdr == nil {
      print("Error: mmap failed, errno = \(errno)")
      return nil
    }

    let numBytes = fileSize / MemoryLayout<Float>.stride
    pointer = hdr.bindMemory(to: Float.self, capacity: numBytes)
    if pointer == UnsafeMutablePointer<Float>(bitPattern: -1) {
      print("Error: mmap failed, errno = \(errno)")
      return nil
    }
  }

  deinit{
    if let hdr = hdr {
      let result = munmap(hdr, Int(fileSize))
      assert(result == 0, "Error: munmap failed, errno = \(errno)")
    }
    if let fd = fd {
      close(fd)
    }
  }
}
