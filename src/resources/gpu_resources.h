#ifndef VULKAN_INTRO_GPU_RESOURCES
#define VULKAN_INTRO_GPU_RESOURCES

#include "vulkan_layer.h"

enum ResourceLocation : unsigned char { Missing = 0, OnDiskBit = 1, ValidBit = 2, InMemoryBit = 4, UploadedBit = 8 };

struct ImageResource {
    unsigned char mLocation = ResourceLocation::Missing;

    uint32_t width = 0, height = 0, channel = 0, mipLevel = 0, componentSize = 0;
    bool sRGB = false, transparent = false;
    vk::Format pixelFormat;

    vk::ImageUsageFlags usage;

    std::vector<std::byte> data;

    VulkanImage image;

    void upload();
    void destroy();

    void calculatePixelFormat(int pixelType);
    void calculateTransparency();
};

struct BufferResource {
    unsigned char mLocation = ResourceLocation::Missing;

    vk::BufferUsageFlags usage;

    std::vector<std::byte> data;

    VulkanBuffer buffer;

    void upload();
    void destroy();
};

struct SamplerResource {
    unsigned char mLocation = ResourceLocation::Missing;

    vk::SamplerCreateInfo info;

    vk::Sampler sampler;

    void upload();
    void destroy();
};

#endif//VULKAN_INTRO_GPU_RESOURCES
