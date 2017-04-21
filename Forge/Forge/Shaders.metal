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

#include <metal_stdlib>
using namespace metal;

kernel void rgb2Gray(
  texture2d<half, access::read> inTexture [[texture(0)]],
  texture2d<half, access::write> outTexture [[texture(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height()) {
      return;
  }
  const half4 inColor = inTexture.read(gid);
  const half y = inColor.x*0.299h + inColor.y*0.587h + inColor.z*0.114h;
  outTexture.write(half4(y * 255.0h, 0.0h, 0.0h, 0.0h), gid);
}

kernel void rgb2bgr(
  texture2d<half, access::read> inTexture [[texture(0)]],
  texture2d<half, access::write> outTexture [[texture(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height()) {
      return;
  }
  const half4 inColor = inTexture.read(gid);
  outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0h), gid);
}

constant half4 meanColor [[ function_constant(0) ]];
constant half meanScale [[ function_constant(1) ]];

kernel void subtractMeanColor(
  texture2d<half, access::read> inTexture [[texture(0)]],
  texture2d<half, access::write> outTexture [[texture(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height()) {
      return;
  }
  outTexture.write(inTexture.read(gid) * meanScale - meanColor, gid);
}
