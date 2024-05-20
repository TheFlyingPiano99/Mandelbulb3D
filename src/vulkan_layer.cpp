#include "vulkan_layer.h"
#include <GLFW/glfw3.h>
#include "constants.h"
#include "window_manager.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace fs = std::filesystem;

namespace vma
{
    VulkanImage createImage(
        VmaAllocator allocator,
        const vk::ImageCreateInfo& imageCreateInfo,
        const VmaAllocationCreateInfo& allocationCreateInfo)
    {
        VulkanImage ret = {};
        VkImage handle;
        VK_CHECK_RESULT(vmaCreateImage(allocator, &(const VkImageCreateInfo&)imageCreateInfo, &allocationCreateInfo, &handle, &ret.memory, &ret.allocationInfo))
        ret.image = handle;
        return ret;
    } 

    VulkanBuffer createBuffer(
        VmaAllocator allocator,
        const vk::BufferCreateInfo& bufferCreateInfo,
        const VmaAllocationCreateInfo& allocationCreateInfo)
    {
        VulkanBuffer ret = {};
        VkBuffer handle;
        VK_CHECK_RESULT(vmaCreateBuffer(allocator, &(const VkBufferCreateInfo&)bufferCreateInfo, &allocationCreateInfo, &handle, &ret.memory, &ret.allocationInfo))
        ret.buffer = handle;
        return ret;
    }
}

void VulkanImage::flush() { vmaFlushAllocation(theVulkanLayer.allocator, memory, 0, VK_WHOLE_SIZE); }
void VulkanImage::invalidate() { vmaInvalidateAllocation(theVulkanLayer.allocator, memory, 0, VK_WHOLE_SIZE); }
void VulkanBuffer::flush() { vmaFlushAllocation(theVulkanLayer.allocator, memory, 0, VK_WHOLE_SIZE); }
void VulkanBuffer::invalidate() { vmaInvalidateAllocation(theVulkanLayer.allocator, memory, 0, VK_WHOLE_SIZE); }
void VulkanCubeImage::flush() { vmaFlushAllocation(theVulkanLayer.allocator, memory, 0, VK_WHOLE_SIZE); }
void VulkanCubeImage::invalidate() { vmaInvalidateAllocation(theVulkanLayer.allocator, memory, 0, VK_WHOLE_SIZE); }

void* VulkanImage::getNativeWin32Handle()
{
    vk::MemoryGetWin32HandleInfoKHR info = {};
    info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
    info.memory = allocationInfo.deviceMemory;
    return theVulkanLayer.device.getMemoryWin32HandleKHR(info);
}

void* VulkanLayer::getNativeWin32Handle(vk::Semaphore semaphore)
{
    vk::SemaphoreGetWin32HandleInfoKHR info = {};
    info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
    info.semaphore = semaphore;
    return theVulkanLayer.device.getSemaphoreWin32HandleKHR(info);
}

void VulkanImage::destroy()
{
    if (view) theVulkanLayer.device.destroy(view); view = nullptr;
    vmaDestroyImage(theVulkanLayer.allocator, image, memory);
    image = nullptr;
    memory = nullptr;
}

void* VulkanImage::map()
{
    void* ret;
    VK_CHECK_RESULT(vmaMapMemory(theVulkanLayer.allocator, memory, &ret))
    return ret;
}

void VulkanImage::unmap()
{
    vmaUnmapMemory(theVulkanLayer.allocator, memory);
}

VulkanImage& VulkanImage::name(const char* pName)
{
    theVulkanLayer.name(image, pName);
    theVulkanLayer.name(view, pName);
    return *this;
}

void VulkanCubeImage::destroy()
{
    theVulkanLayer.device.destroy(view); view = nullptr;
    for (auto& it : face) {
        theVulkanLayer.device.destroy(it);
        it = nullptr;
    }
    vmaDestroyImage(theVulkanLayer.allocator, image, memory);
    image = nullptr;
    memory = nullptr;
}

void* VulkanCubeImage::map()
{
    void* ret;
    VK_CHECK_RESULT(vmaMapMemory(theVulkanLayer.allocator, memory, &ret))
    return ret;
}

void VulkanCubeImage::unmap()
{
    vmaUnmapMemory(theVulkanLayer.allocator, memory);
}

VulkanCubeImage& VulkanCubeImage::name(const char* pName)
{
    theVulkanLayer.name(image, pName);
    theVulkanLayer.name(view, pName);
    for (auto& it : face) {
        theVulkanLayer.name(it, pName);
    }
    return *this;
}

void VulkanBuffer::destroy()
{
    vmaDestroyBuffer(theVulkanLayer.allocator, buffer, memory);
    buffer = nullptr;
    memory = nullptr;
}

void* VulkanBuffer::map()
{
    void* ret;
    VK_CHECK_RESULT(vmaMapMemory(theVulkanLayer.allocator, memory, &ret))
    return ret;
}

void VulkanBuffer::unmap()
{
    vmaUnmapMemory(theVulkanLayer.allocator, memory);
}

VulkanBuffer& VulkanBuffer::name(const char* pName)
{
    theVulkanLayer.name(buffer, pName);
    return *this;
}

void VulkanLayer::initializeVulkan()
{
    glfwInit();
    createInstance(cDebugValidationLayers);
    createDebugMessenger();
}

void VulkanLayer::initializeDevices(const vk::SurfaceKHR& surface)
{
    pSurface = &surface;
    pickPhysicalDevice();
    collectDeviceExtensions();
    createLogicalDevices();
    createCommandPool();
    createPipelineCache();
}

void VulkanLayer::destroyVulkan()
{
    if (messenger) instance.destroyDebugUtilsMessengerEXT(messenger);
    instance.destroy();
    glfwTerminate();
}

void VulkanLayer::destroyDevices()
{
    writePipelineCache();
    device.destroy(pipelineCache);
    device.destroy(commandPool);
    device.destroy(transientCommandPool);
    device.destroy(transferCommandPool);
    vmaDestroyAllocator(allocator);
    device.destroy();
}

vk::ImageView VulkanLayer::createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
{
    vk::ImageViewCreateInfo viewInfo = {};
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    return device.createImageView(viewInfo);
}

VulkanImage VulkanLayer::createVolume(uint32_t width, uint32_t height, uint32_t depth, vk::Format format, vk::ImageLayout layout, vk::ImageUsageFlags usage, VmaAllocationCreateInfo memoryInfo, vk::ImageCreateFlags flags, vk::ImageTiling tiling)
{
    std::array<uint32_t, 2> queues = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.transferFamily.value() };

    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e3D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = depth;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.queueFamilyIndexCount = queues.size();
    imageInfo.pQueueFamilyIndices = queues.data();
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.flags = flags;

    auto ret = vma::createImage(allocator, imageInfo, memoryInfo);

    vk::ImageViewCreateInfo viewInfo = {};
    viewInfo.image = ret.image;
    viewInfo.viewType = vk::ImageViewType::e3D;
    viewInfo.format = format;
    viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    ret.view = device.createImageView(viewInfo);
    ret.format = format;
    ret.width = width; ret.height = height;

    if (layout != vk::ImageLayout::eUndefined) {
        auto cmd = beginSingleTimeTransferCommand();
        vk::ImageMemoryBarrier2 bar = {};
        bar.image = ret.image;
        bar.oldLayout = vk::ImageLayout::eUndefined;
        bar.newLayout = layout;
        bar.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        bar.dstStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        bar.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = 1;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = 1;

        vk::DependencyInfo dep = {};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;

        cmd.pipelineBarrier2(dep);
        endSingleTimeTransferCommand(cmd);
    }

    return ret;
}

VulkanImage VulkanLayer::createVolume(uint32_t width, uint32_t height, uint32_t depth, vk::Format format, vk::ImageLayout layout, vk::ImageUsageFlags usage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags, vk::ImageCreateFlags flags, vk::ImageTiling tiling)
{
    VmaAllocationCreateInfo info = {};
    info.usage = (VmaMemoryUsage)memoryUsage;
    info.flags = allocationFlags;
    return createVolume(width, height, depth, format, layout, usage, info, flags, tiling);
}

VulkanImage VulkanLayer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSample, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, VmaAllocationCreateInfo memoryInfo, vk::ImageTiling tiling, void* pNext)
{
    std::array<uint32_t, 2> queues = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.transferFamily.value() };

    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.queueFamilyIndexCount = queues.size();
    imageInfo.pQueueFamilyIndices = queues.data();
    imageInfo.samples = numSample;
    imageInfo.pNext = pNext;

    auto ret = vma::createImage(allocator, imageInfo, memoryInfo);
    ret.format = format;
    ret.width = width; ret.height = height;

    // if only transfer usage, then we cant create image view (and we dont need to)
    if ((imageInfo.usage & ~(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc)) != vk::ImageUsageFlags{}) {
        ret.view = createImageView(ret.image, format, aspect, mipLevels);
    }

    if (layout != vk::ImageLayout::eUndefined) {
        auto cmd = beginSingleTimeTransferCommand();
        vk::ImageMemoryBarrier2 bar = {};
        bar.image = ret.image;
        bar.oldLayout = vk::ImageLayout::eUndefined;
        bar.newLayout = layout;
        bar.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        bar.dstStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        bar.subresourceRange.aspectMask = aspect;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = 1;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = mipLevels;

        vk::DependencyInfo dep = {};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;

        cmd.pipelineBarrier2(dep);
        endSingleTimeTransferCommand(cmd);
    }

    return ret;
}

VulkanImage VulkanLayer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSample, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags,  vk::ImageTiling tiling, void* pNext)
{
    VmaAllocationCreateInfo info = {};
    info.usage = (VmaMemoryUsage)memoryUsage;
    info.flags = allocationFlags;
    return createImage(width, height, mipLevels, numSample, format, layout, aspect, usage, info, tiling, pNext);
}

VulkanCubeImage VulkanLayer::createCubeMap(uint32_t width, uint32_t height, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, VmaAllocationCreateInfo memoryInfo, vk::ImageTiling tiling)
{
    std::array<uint32_t, 2> queues = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.transferFamily.value() };

    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 6;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.queueFamilyIndexCount = queues.size();
    imageInfo.pQueueFamilyIndices = queues.data();
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;

    auto img = vma::createImage(allocator, imageInfo, memoryInfo);
    VulkanCubeImage ret;
    ret.image = img.image;
    ret.memory = img.memory;

    vk::ImageViewCreateInfo viewInfo = {};
    viewInfo.image = ret.image;
    viewInfo.viewType = vk::ImageViewType::eCube;
    viewInfo.format = format;
    viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
    viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
    viewInfo.subresourceRange.aspectMask = aspect;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    ret.view = device.createImageView(viewInfo);

    for (uint32_t i = 0; i < 6; ++i) {
        vk::ImageViewCreateInfo faceViewInfo = {};
        faceViewInfo.image = ret.image;
        faceViewInfo.viewType = vk::ImageViewType::e2D;
        faceViewInfo.format = format;
        faceViewInfo.components.r = vk::ComponentSwizzle::eIdentity;
        faceViewInfo.components.g = vk::ComponentSwizzle::eIdentity;
        faceViewInfo.components.b = vk::ComponentSwizzle::eIdentity;
        faceViewInfo.components.a = vk::ComponentSwizzle::eIdentity;
        faceViewInfo.subresourceRange.aspectMask = aspect;
        faceViewInfo.subresourceRange.baseMipLevel = 0;
        faceViewInfo.subresourceRange.levelCount = 1;
        faceViewInfo.subresourceRange.baseArrayLayer = i;
        faceViewInfo.subresourceRange.layerCount = 1;

        ret.face[i] = device.createImageView(faceViewInfo);
    }

    if (layout != vk::ImageLayout::eUndefined) {
        auto cmd = beginSingleTimeTransferCommand();
        vk::ImageMemoryBarrier2 bar = {};
        bar.image = ret.image;
        bar.oldLayout = vk::ImageLayout::eUndefined;
        bar.newLayout = layout;
        bar.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        bar.dstStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        bar.subresourceRange.aspectMask = aspect;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = 1;

        vk::DependencyInfo dep = {};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;

        cmd.pipelineBarrier2(dep);
        endSingleTimeTransferCommand(cmd);
    }

    return ret;
}

VulkanCubeImage VulkanLayer::createCubeMap(uint32_t width, uint32_t height, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags, vk::ImageTiling tiling)
{
    VmaAllocationCreateInfo info = {};
    info.usage = (VmaMemoryUsage)memoryUsage;
    info.flags = allocationFlags;
    return createCubeMap(width, height, format, layout, aspect, usage, info, tiling);
}

VulkanBuffer VulkanLayer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, VmaAllocationCreateInfo memoryInfo)
{
    std::array<uint32_t, 2> queues = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.transferFamily.value() };

    vk::BufferCreateInfo createInfo = {};
    createInfo.size = size;
    createInfo.usage = usage;
    createInfo.sharingMode = vk::SharingMode::eExclusive;
    createInfo.queueFamilyIndexCount = queues.size();
    createInfo.pQueueFamilyIndices = queues.data();

    return vma::createBuffer(allocator, createInfo, memoryInfo);
}

VulkanBuffer VulkanLayer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags)
{
    VmaAllocationCreateInfo info = {};
    info.usage = (VmaMemoryUsage)memoryUsage;
    info.flags = allocationFlags;
    return createBuffer(size, usage, info);
}

VulkanBuffer VulkanLayer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vma::AllocationCreateFlags allocationFlags)
{
    VmaAllocationCreateInfo info = {};
    info.usage = VMA_MEMORY_USAGE_AUTO;
    info.flags = allocationFlags;
    return createBuffer(size, usage, info);
}

// TODO: implement pooling
VulkanBuffer VulkanLayer::requestStagingBuffer(vk::DeviceSize size)
{
    return createBuffer(size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vma::MemoryUsage::eAuto,
        vma::AllocationCreateFlagBits::eCreateMappedBit | vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit
    );
}

void VulkanLayer::returnStagingBuffer(VulkanBuffer& buffer)
{
    buffer.destroy();
}

vk::PipelineLayout VulkanLayer::createPipelineLayout(std::initializer_list<vk::DescriptorSetLayout> descriptorSets, uint32_t pushConstantRange, vk::ShaderStageFlags pushConstantFlags)
{
    vk::PushConstantRange pcr = {};
    pcr.stageFlags = pushConstantFlags;
    pcr.offset = 0;
    pcr.size = pushConstantRange;

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.setLayoutCount = (uint32_t)descriptorSets.size();
    pipelineLayoutInfo.pSetLayouts = descriptorSets.begin();
    pipelineLayoutInfo.pushConstantRangeCount = pcr.size == 0 ? 0 : 1;
    pipelineLayoutInfo.pPushConstantRanges = pcr.size == 0 ? nullptr : &pcr;

    return device.createPipelineLayout(pipelineLayoutInfo);
}

vk::Pipeline VulkanLayer::createComputePipeline(vk::ShaderModule computeShader, vk::PipelineLayout pipelineLayout)
{
    vk::PipelineShaderStageCreateInfo compShaderStageInfo = {};
    compShaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
    compShaderStageInfo.module = computeShader;
    compShaderStageInfo.pName = "main";
    compShaderStageInfo.pSpecializationInfo = nullptr;

    vk::ComputePipelineCreateInfo info = {};
    info.layout = pipelineLayout;
    info.stage = compShaderStageInfo;
    info.flags = {};

    return device.createComputePipeline(pipelineCache, info).value;
}

vk::Pipeline VulkanLayer::createGraphicsPipeline(vk::GraphicsPipelineCreateInfo createInfo)
{
    return device.createGraphicsPipeline(pipelineCache, createInfo).value;
}

void VulkanLayer::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
    auto cmdBuffer = beginSingleTimeTransferCommand();

    vk::BufferCopy copyRegion = {};
    copyRegion.srcOffset = copyRegion.dstOffset = 0;
    copyRegion.size = size;
    cmdBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeTransferCommand(cmdBuffer);
}

vk::CommandBuffer VulkanLayer::beginSingleTimeCommands()
{
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = transientCommandPool;
    allocInfo.commandBufferCount = 1;

    auto cmdBuffer = device.allocateCommandBuffers(allocInfo);

    vk::CommandBufferBeginInfo beginInfo = {};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    cmdBuffer[0].begin(beginInfo);
    return cmdBuffer[0];
}

void VulkanLayer::endSingleTimeCommands(vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo = {};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VK_CHECK_RESULT(safeSubmitGraphicsCommand(submitInfo, true))
    device.freeCommandBuffers(transientCommandPool, commandBuffer);
}

vk::CommandBuffer VulkanLayer::beginSingleTimeTransferCommand()
{
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = transferCommandPool;
    allocInfo.commandBufferCount = 1;

    auto cmdBuffer = device.allocateCommandBuffers(allocInfo);

    vk::CommandBufferBeginInfo beginInfo = {};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    cmdBuffer[0].begin(beginInfo);
    return cmdBuffer[0];
}

void VulkanLayer::endSingleTimeTransferCommand(vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo = {};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VK_CHECK_RESULT(safeSubmitTransferCommand(submitInfo, true))
    device.freeCommandBuffers(transferCommandPool, commandBuffer);
}

vk::Result VulkanLayer::safeSubmitPresentCommand(vk::PresentInfoKHR presentInfo, bool wait)
{
    std::unique_lock<std::mutex> lock{ mQueueMutex };
    auto res = presentQueue.presentKHR(presentInfo);
    if (wait) presentQueue.waitIdle();
    return res;
}

vk::Result VulkanLayer::safeSubmitGraphicsCommand(vk::SubmitInfo submitInfo, bool wait, vk::Fence fence)
{
    std::unique_lock<std::mutex> lock{ mQueueMutex };
    auto res = graphicsQueue.submit(1, &submitInfo, fence);
    if (wait) graphicsQueue.waitIdle();
    return res;
}

vk::Result VulkanLayer::safeSubmitTransferCommand(vk::SubmitInfo submitInfo, bool wait, vk::Fence fence)
{
    std::unique_lock<std::mutex> lock{ mQueueMutex };
    auto res = transferQueue.submit(1, &submitInfo, fence);
    if (wait) transferQueue.waitIdle();
    return res;
}

void VulkanLayer::safeWaitIdle()
{
    std::unique_lock<std::mutex> lock{ mQueueMutex };
    device.waitIdle();
}

void VulkanLayer::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height, uint32_t depth)
{
    auto commandBuffer = beginSingleTimeTransferCommand();

    vk::BufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = vk::Offset3D{0, 0, 0};
    region.imageExtent = vk::Extent3D{width, height, depth};

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    endSingleTimeTransferCommand(commandBuffer);
}

vk::Format VulkanLayer::findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
{
    for (auto format : candidates) {
        auto props = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features){
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features){
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

vk::Format VulkanLayer::findDepthFormat()
{
    return findSupportedFormat(
            {vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

void VulkanLayer::generateMipmaps(vk::Image image, vk::Format imageFormat, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels) {
    auto formatProps = physicalDevice.getFormatProperties(imageFormat);
    if (!(formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    auto commandBuffer = beginSingleTimeTransferCommand();

    vk::ImageMemoryBarrier barrier = {};
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = (int32_t)texWidth;
    int32_t mipHeight = (int32_t)texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlags{},
                0, nullptr, 0, nullptr,
                1, &barrier);

        vk::ImageBlit blit = {};
        blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
        blit.srcOffsets[1] = vk::Offset3D{ mipWidth, mipHeight, 1 };
        blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
        blit.dstOffsets[1] = vk::Offset3D{ mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
        blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
                                      vk::DependencyFlags{},
                                      0, nullptr, 0, nullptr,
                                      1, &barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::DependencyFlags{},
                                  0, nullptr, 0, nullptr,
                                  1, & barrier);

    endSingleTimeTransferCommand(commandBuffer);
}

void VulkanLayer::createInstance(bool enableValidationLayers)
{
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    isValidationEnabled = enableValidationLayers;
    collectRequiredInstanceExtensions(enableValidationLayers);
    if (!checkRequiredExtensionSupport()){
        throw std::runtime_error("required extensions are not supported!");
    }

    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "Vulkan App";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    vk::InstanceCreateInfo createInfo;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    instance = vk::createInstance(createInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
}

void VulkanLayer::createDebugMessenger()
{
    vk::DebugUtilsMessengerCreateInfoEXT createInfo = {};

    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral;

    createInfo.pfnUserCallback = [] (auto messageSeverity, auto /* messageType */, auto pCallbackData, void* /* pUserData */) -> unsigned  int {
        if (messageSeverity == (int)vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
            vulkanLogger.LogError(pCallbackData->pMessage);
        } else if (messageSeverity == (int)vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            vulkanLogger.LogWarning(pCallbackData->pMessage);
        } else if (messageSeverity == (int)vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
            vulkanLogger.LogInfo(pCallbackData->pMessage);
        }
        return VK_FALSE;
    };
    createInfo.pUserData = nullptr;

    if (isValidationEnabled) {
        messenger = instance.createDebugUtilsMessengerEXT(createInfo);
    }
}

void VulkanLayer::name_impl(vk::ObjectType objectType, uint64_t handle, const char* pObjectName)
{
    if (cDebugObjectNames) {
        vk::DebugUtilsObjectNameInfoEXT info = {};
        info.objectType = objectType;
        info.objectHandle = handle;
        info.pObjectName = pObjectName;
        device.setDebugUtilsObjectNameEXT(info);
    }
}

void VulkanLayer::pickPhysicalDevice()
{
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (const auto& dev : devices) {
        if (isDeviceSuitable(dev)) {
            physicalDevice = dev;
            queueFamilyIndices = findQueueFamilies(physicalDevice);
            maxMSAASamples = getMaxUsableSampleCount();
            physicalDeviceProperties = dev.getProperties();
            break;
        }
    }

    if (!physicalDevice) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VulkanLayer::createLogicalDevices()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value(), indices.transferFamily.value()};

    float queuePriority[] = {1.0f};
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceMaintenance4Features maintance4Features = {};
    maintance4Features.maintenance4 = VK_TRUE;

    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeature = {};
    dynamicRenderingFeature.dynamicRendering = VK_TRUE;
    dynamicRenderingFeature.pNext = &maintance4Features;

    vk::PhysicalDeviceDescriptorIndexingFeatures indexingFeatures = {};
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    indexingFeatures.pNext = &dynamicRenderingFeature;

    vk::PhysicalDeviceSynchronization2FeaturesKHR synchFeatures = {};
    synchFeatures.synchronization2 = VK_TRUE;
    synchFeatures.pNext = &indexingFeatures;

    vk::PhysicalDeviceFeatures2 deviceFeatures = {};
    deviceFeatures.features.samplerAnisotropy = VK_TRUE;
    deviceFeatures.features.fragmentStoresAndAtomics = VK_TRUE;
    deviceFeatures.pNext = &synchFeatures;

    vk::DeviceCreateInfo createInfo = {};
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = nullptr;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
    createInfo.pNext = &deviceFeatures;

    device = physicalDevice.createDevice(createInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.flags = {};

    if (memoryBudgetAvailable) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }    

    vmaCreateAllocator(&allocatorInfo, &allocator);

    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    graphicsFamilyIndex = indices.graphicsFamily.value();
    presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    transferQueue = graphicsQueue;
}

void VulkanLayer::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo = {};
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    commandPool = device.createCommandPool(poolInfo);

    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
    transientCommandPool = device.createCommandPool(poolInfo);

    poolInfo.queueFamilyIndex = queueFamilyIndices.transferFamily.value();
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
    transferCommandPool = device.createCommandPool(poolInfo);
}

void VulkanLayer::createPipelineCache()
{
    std::vector<char> data;

    try {
        data = readFile(cDirCache / "pipelineCache.dat");
    } catch (...) {}

    vk::PipelineCacheCreateInfo createInfo;
    createInfo.initialDataSize = data.size();
    createInfo.pInitialData = data.data();

    pipelineCache = device.createPipelineCache(createInfo);
}

void VulkanLayer::writePipelineCache()
{
    auto data = device.getPipelineCacheData(pipelineCache);
    writeFile(cDirCache / "pipelineCache.dat", data);
}

bool VulkanLayer::checkDeviceExtensionSupport(const vk::PhysicalDevice& dev)
{
    auto availableExtensions = dev.enumerateDeviceExtensionProperties();
    for (auto extension : deviceExtensions){
        bool extensionFound = false;
        for (const auto& aExt : availableExtensions) {
            if (strcmp(extension, aExt.extensionName) == 0) {
                extensionFound = true;
                break;
            }
        }

        if (!extensionFound) {
            return false;
        }
    }
    
    for (const auto& aExt : availableExtensions) {
        if (strcmp(aExt.extensionName, VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME) == 0) {
            swapChainMutableFormatAvailable = true;
        } else if (strcmp(aExt.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            memoryBudgetAvailable = true;
        }
    }

    return true;
}

void VulkanLayer::collectRequiredInstanceExtensions(bool enableValidationLayers)
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    for (auto i : rs::iota(0u, glfwExtensionCount)){
        instanceExtensions.push_back(glfwExtensions[i]);
    }

    if (enableValidationLayers) {
        validationLayers.push_back("VK_LAYER_KHRONOS_validation");

        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
}

void VulkanLayer::collectDeviceExtensions()
{
    std::set<std::string> exts;
    auto props = physicalDevice.enumerateDeviceExtensionProperties();
    for (auto& e : props) {
        exts.insert(e.extensionName);
    }

    // mandatory
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);

    if (swapChainMutableFormatAvailable) {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME);
    }
    if (memoryBudgetAvailable) {
        deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
}

bool VulkanLayer::checkRequiredExtensionSupport()
{
    auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    for (const char* layerName : instanceExtensions) {
        bool extensionsFound = false;
        for (const auto& extensionProperties : availableExtensions) {
            if (strcmp(layerName, extensionProperties.extensionName) == 0) {
                extensionsFound = true;
                break;
            }
        }

        if (!extensionsFound) {
            return false;
        }
    }
    return true;
}

bool VulkanLayer::checkValidationLayerSupport()
{
    auto availableLayers =  vk::enumerateInstanceLayerProperties();
    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }
    return true;
}

bool VulkanLayer::isDeviceSuitable(const vk::PhysicalDevice& dev)
{
    QueueFamilyIndices indices = findQueueFamilies(dev);

    bool extensionsSupported = checkDeviceExtensionSupport(dev);

    bool swapChainAdequate = false;
    auto swapChainSupportDetails = theWindowManager.querySwapChainSupport(dev);
    if (extensionsSupported) {
        swapChainAdequate = !swapChainSupportDetails.formats.empty() && !swapChainSupportDetails.presentModes.empty();
    }

    auto supportedFeatures = dev.getFeatures();

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy
                                && supportedFeatures.fragmentStoresAndAtomics && supportedFeatures.vertexPipelineStoresAndAtomics;
}

vk::SampleCountFlagBits VulkanLayer::getMaxUsableSampleCount() {
    auto props = physicalDevice.getProperties();

    timestampPeriod = props.limits.timestampPeriod;

    vk::SampleCountFlags counts = vk::SampleCountFlags{std::min((unsigned int)props.limits.framebufferColorSampleCounts, (unsigned int)props.limits.framebufferDepthSampleCounts)};
    if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
    if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
    if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

    return vk::SampleCountFlagBits::e1;
}


VulkanLayer::QueueFamilyIndices VulkanLayer::findQueueFamilies(const vk::PhysicalDevice& dev)
{
    QueueFamilyIndices indices;
    auto queueFamilies = dev.getQueueFamilyProperties();

    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
            indices.transferFamily = i;
        }

        if (queueFamily.queueCount > 0 && dev.getSurfaceSupportKHR(i, *pSurface)) {
            indices.presentFamily = i;
        }

       /* if (queueFamily.queueCount > 0 && !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) && queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) {
            indices.transferFamily = i;
        }*/

        if (indices.isComplete()) {
            break;
        }

        i++;
    }
    return indices;
}

vk::ShaderModule VulkanLayer::createShaderModule(const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo = {};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    return device.createShaderModule(createInfo);
}

std::vector<char> VulkanLayer::readFile(const std::filesystem::path& file)
{
    return readFile(file.string());
}

std::vector<char> VulkanLayer::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        vulkanLogger.LogError("Failed to open ", filename, " for reading.");
        return std::vector<char>(4, 0);
    }

    std::streamsize fileSize = file.tellg();
    std::vector<char> buffer((size_t)fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

vk::PipelineShaderStageCreateInfo VulkanLayer::loadShaderStageCreateInfo(const std::string& path, vk::ShaderStageFlagBits stage) {
    vk::PipelineShaderStageCreateInfo info = {};
    info.stage = stage;
    info.module = createShaderModule(readFile(path));
    info.pName = "main";
    info.pSpecializationInfo = nullptr;
    return info;
}

void VulkanLayer::writeFile(const std::filesystem::path& file, const std::vector<uint8_t>& data)
{
    writeFile(file.string(), data);
}

void VulkanLayer::writeFile(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream file(filename, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!file.is_open()) {
        vulkanLogger.LogError("Failed to open ", filename, " for writing.");
        return;
    }

    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(uint8_t)));
    file.close();
}
