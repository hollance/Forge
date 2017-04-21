#include <metal_stdlib>
using namespace metal;

kernel void preprocessMNIST(
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

  // Convert white to black and black to white. The MNIST network is trained
  // on white-on-black images but we want to detect black-on-white digits, as
  // they are more common (unless you write on a blackboard...)
  y = 1.0h - y;

  // Increase the contrast. There are a few different methods:

  //if (y < 0.5h) y = 0.0h; else y = 1.0h;

  //y = y*y;

  //y = sin(3.14h * (y - 0.5h))/2.0h + 0.5h;

  // Using a fairly extreme sigmoid function seems to work best. Note that
  // the recognition does not work well if part of the image is in shadow
  // and part in bright light. The background color should be as equal as
  // possible everywhere.
  y = (y - 0.5h) * 100.0h;
  y = 1.0h / (1.0h + exp(-y));

  // Only write into the first color channel.
  half4 outColor = half4(y * 255.0h, 0.0h, 0.0h, 0.0h);

  outTexture.write(outColor, gid);
}
