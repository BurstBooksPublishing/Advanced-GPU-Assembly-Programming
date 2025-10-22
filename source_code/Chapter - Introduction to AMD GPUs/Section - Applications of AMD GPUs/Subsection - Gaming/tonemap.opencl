__kernel void tonemap_tiled(__read_only image2d_t srcImg, // input HDR texture
                            __write_only image2d_t dstImg, // output LDR texture
                            float exposure) {
  // local tile size: 8x8 pixels; adjust to fit LDS and wave size
  __local float4 tile[8][8]; // LDS accumulator for tile (fits in shared memory)

  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  const int2 coords = (int2)(gx, gy);

  // Vectorized load from texture (coalesced if workgroup layout matches)
  float4 hdr = read_imagef(srcImg, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP, coords);

  // Local accumulation: write into LDS so nearby threads reuse working set
  tile[ly][lx] = hdr;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Simple tone mapping: compute average luminance of tile then apply exposure
  // Only one thread per tile computes average to avoid extra barriers
  float lum = 0.0f;
  if (lx == 0 && ly == 0) {
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j)
      for (int i = 0; i < 8; ++i) {
        float3 c = vload3(0, &tile[j][i].x); // extract RGB
        sum += 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
      }
    lum = sum / 64.0f;
    // Broadcast average luminance to tile via LDS slot (reuse tile[0][0])
    tile[0][0].x = lum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Read broadcasted luminance and apply filmic tone map
  lum = tile[0][0].x;
  float4 c = tile[ly][lx];
  float4 mapped = c * (exposure / (lum + 1e-6f));
  // Simple gamma
  mapped.xyz = pow(mapped.xyz, (float3)(1.0f/2.2f));

  write_imagef(dstImg, coords, mapped);
}