struct ScopedLabel { // RAII helper to mark command buffer regions for RGP
    VkCommandBuffer cmd;
    VkDebugUtilsLabelEXT labelInfo;
    PFN_vkCmdBeginDebugUtilsLabelEXT vkBegin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT vkEnd = nullptr;
    ScopedLabel(VkCommandBuffer cb, VkDevice dev, const char* name, float color[4]) : cmd(cb) {
        // resolve extension pointers (must be available if VK_EXT_debug_utils enabled)
        vkBegin = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(dev, "vkCmdBeginDebugUtilsLabelEXT");
        vkEnd   = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(dev, "vkCmdEndDebugUtilsLabelEXT");
        // prepare label (UTF-8 null-terminated)
        labelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        labelInfo.pNext = nullptr;
        labelInfo.pLabelName = name;
        memcpy(labelInfo.color, color, sizeof(labelInfo.color));
        if (vkBegin) vkBegin(cmd, &labelInfo); // begin label (visible to RGP)
    }
    ~ScopedLabel() { if (vkEnd) vkEnd(cmd); } // end label automatically
};