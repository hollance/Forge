#include <metal_stdlib>
using namespace metal;

kernel void preprocess(
  texture2d<half, access::read> inTexture [[texture(0)]],
  texture2d<half, access::write> outTexture [[texture(1)]],
  uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height()) {
      return;
  }

  // Subtract mean values, scale by 0.017, convert to BGR.

  const auto means = float4(123.68f, 116.78f, 103.94f, 0.0f);
  const auto inColor = (float4(inTexture.read(gid)) * 255.0f - means) * 0.017f;
  outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}
