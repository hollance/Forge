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

  half4 inColor = inTexture.read(gid);

  // Convert to grayscale.
  half y = inColor.x*0.299h + inColor.y*0.587h + inColor.z*0.114h;

  // Only write into the first color channel.
  half4 outColor = half4(y * 255.0h, 0.0h, 0.0h, 0.0h);

  outTexture.write(outColor, gid);
}
