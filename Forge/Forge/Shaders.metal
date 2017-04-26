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

constant half4 meanColor [[ function_constant(0) ]];
constant half meanScale [[ function_constant(1) ]];
constant ushort2 imageStride [[ function_constant(2) ]];
constant bool applyReLU [[ function_constant(3) ]];

// MARK: - Preprocessing kernels

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

// MARK: - Depth-wise convolution

kernel void depthwiseConv3x3_half(
  texture2d<half, access::sample> inTexture [[texture(0)]],
  texture2d<half, access::write> outTexture [[texture(1)]],
  const device half4* weights [[buffer(0)]],
  ushort2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height()) return;

  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);

  // Note: this is a very naive implementation of convolution.
  // There are ways to make it a lot faster...

  // Seen from the destination image, the stride is how far apart the pixels
  // are in the source image.
  const ushort2 pos = gid * imageStride;

  // Read the 3x3 pixels surrounding the source pixel.
  // By processing the pixels as half4 values we do up to 4 channels at a time.
  half4 in[9];
  in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1));
  in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1));
  in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1));
  in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ));
  in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ));
  in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ));
  in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1));
  in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1));
  in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1));

  // Multiply by the weights and put the weighted sum in the output pixel.
  half4 out = half4(0.0h);
  for (ushort t = 0; t < 9; ++t) {
    out += in[t] * weights[t];
  }

  // Applying a ReLU in the shader is quicker than creating a new layer for it.
  if (applyReLU) out = fmax(out, 0.0h);

  outTexture.write(out, gid);
}

kernel void depthwiseConv3x3_half_array(
  texture2d_array<half, access::sample> inTexture [[texture(0)]],
  texture2d_array<half, access::write> outTexture [[texture(1)]],
  const device half4* weights [[buffer(0)]],
  ushort3 gid [[thread_position_in_grid]])
{
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;

  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);

  const ushort2 pos = gid.xy * imageStride;
  const ushort slices = outTexture.get_array_size();
  const ushort slice = gid.z;

  half4 in[9];
  in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), slice);
  in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), slice);
  in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), slice);
  in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), slice);
  in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), slice);
  in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), slice);
  in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), slice);
  in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), slice);
  in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), slice);

  half4 out = half4(0.0h);
  for (ushort t = 0; t < 9; ++t) {
    out += in[t] * weights[t*slices + slice];
  }

  if (applyReLU) out = fmax(out, 0.0h);

  outTexture.write(half4(slices, slice, 0.0h, 0.0h), gid.xy, gid.z);
}
