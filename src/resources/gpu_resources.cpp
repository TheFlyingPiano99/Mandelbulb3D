#include "gpu_resources.h"
#include "vulkan_layer.h"

void ImageResource::upload()
{
    auto& vl = theVulkanLayer;
    if (mLocation & ResourceLocation::UploadedBit) return;

    vk::DeviceSize imageSize = width * height * channel;

    auto imageUsage = usage | vk::ImageUsageFlagBits::eTransferDst;
    if (mipLevel > 1) {
        imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    }

    image = vl.createImage(
        width, height, mipLevel,
        vk::SampleCountFlagBits::e1,
        pixelFormat, vk::ImageLayout::eTransferDstOptimal, vk::ImageAspectFlagBits::eColor, imageUsage
    );

    VkMemoryPropertyFlags props;
    vmaGetAllocationMemoryProperties(theVulkanLayer.allocator, image.memory, &props);
    if (props & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        void* pData = nullptr;
        vmaMapMemory(theVulkanLayer.allocator, image.memory, &pData);
        memcpy(pData, data.data(), imageSize);
        vmaFlushAllocation(theVulkanLayer.allocator, image.memory, 0, VK_WHOLE_SIZE);
        vmaUnmapMemory(theVulkanLayer.allocator, image.memory);
    } else {
        auto stagingBuffer = vl.requestStagingBuffer(imageSize);
        memcpy(stagingBuffer.allocationInfo.pMappedData, data.data(), imageSize);
        stagingBuffer.flush();
        vl.copyBufferToImage(stagingBuffer.buffer, image.image, width, height);
        vl.returnStagingBuffer(stagingBuffer);
    }

    if (mipLevel > 1) {
        vl.generateMipmaps(image.image, pixelFormat, width, height, mipLevel);
    } else {
        auto cmd = vl.beginSingleTimeTransferCommand();
        vk::ImageMemoryBarrier2 bar = {};
        bar.image = image.image;
        bar.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        bar.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        bar.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        bar.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        bar.dstStageMask = vk::PipelineStageFlagBits2::eAllGraphics;
        bar.dstAccessMask = vk::AccessFlagBits2KHR::eShaderRead;
        bar.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = 1;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = 1;

        vk::DependencyInfo dep = {};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;

        cmd.pipelineBarrier2(dep);
        vl.endSingleTimeTransferCommand(cmd);
    }

    mLocation |= ResourceLocation::UploadedBit;
}

void ImageResource::destroy()
{
    if (!(mLocation & ResourceLocation::UploadedBit)) return;

    image.destroy();

    mLocation &= ~ResourceLocation::UploadedBit;
}

void BufferResource::upload()
{
    auto& vl = theVulkanLayer;
    if ((mLocation & ResourceLocation::UploadedBit)) return;

    vk::DeviceSize size = data.size();

    buffer = vl.createBuffer(
        size,
        usage | vk::BufferUsageFlagBits::eTransferDst
    );
    
    VkMemoryPropertyFlags props;
    vmaGetAllocationMemoryProperties(theVulkanLayer.allocator, buffer.memory, &props);
    if (props & VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        void* pData = nullptr;
        vmaMapMemory(theVulkanLayer.allocator, buffer.memory, &pData);
        memcpy(pData, data.data(), size);
        vmaFlushAllocation(theVulkanLayer.allocator, buffer.memory, 0, VK_WHOLE_SIZE);
        vmaUnmapMemory(theVulkanLayer.allocator, buffer.memory);
    } else {
        auto stagingBuffer = vl.requestStagingBuffer(size);
        memcpy(stagingBuffer.allocationInfo.pMappedData, data.data(), size);
        stagingBuffer.flush();
        vl.copyBuffer(stagingBuffer.buffer, buffer.buffer, size);
        vl.returnStagingBuffer(stagingBuffer);
    }

    mLocation |= ResourceLocation::UploadedBit;
}

void BufferResource::destroy()
{
    if (!(mLocation & ResourceLocation::UploadedBit)) return;

    buffer.destroy();

    mLocation &= ~ResourceLocation::UploadedBit;
}

void SamplerResource::upload()
{
    if ((mLocation & ResourceLocation::UploadedBit)) return;

    sampler = theVulkanLayer.device.createSampler(info);

    mLocation |= ResourceLocation::UploadedBit;
}

void SamplerResource::destroy()
{
    if (!(mLocation & ResourceLocation::UploadedBit)) return;

    theVulkanLayer.device.destroy(sampler);

    mLocation &= ~ResourceLocation::UploadedBit;
}
