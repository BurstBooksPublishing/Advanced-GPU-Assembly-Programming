VkStridedDeviceAddressRegionKHR sbtRaygen = { /* deviceAddress/size/stride */ };
VkStridedDeviceAddressRegionKHR sbtMiss   = { /* ... */ };
VkStridedDeviceAddressRegionKHR sbtHit    = { /* ... */ };
VkStridedDeviceAddressRegionKHR sbtCallable= { /* ... */ };

// Record trace call into command buffer (portable Vulkan KHR call).
// Driver maps this to RT cores (NVIDIA) or Ray Accelerators (AMD).
vkCmdTraceRaysKHR(cmdBuffer,
    &sbtRaygen,   // raygen SBT region
    &sbtMiss,     // miss SBT region
    &sbtHit,      // hit SBT region
    &sbtCallable, // callable SBT region
    width, height, 1);