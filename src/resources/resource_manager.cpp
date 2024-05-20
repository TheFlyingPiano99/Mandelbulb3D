#include "resource_manager.hpp"

void ResourceManager::initialize()
{
    SamplerResource samp;
    samp.info.magFilter = samp.info.minFilter = vk::Filter::eLinear;
    samp.info.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samp.info.addressModeU = vk::SamplerAddressMode::eRepeat;
    samp.info.addressModeV = vk::SamplerAddressMode::eRepeat;
    samp.info.addressModeW = vk::SamplerAddressMode::eRepeat;
    samp.info.mipLodBias = 0.f;
    samp.info.anisotropyEnable = VK_TRUE;
    samp.info.maxAnisotropy = 16.f;
    samp.info.compareEnable = VK_FALSE;
    samp.info.minLod = 0.f;
    samp.info.maxLod = 200.f;
    samp.info.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samp.info.unnormalizedCoordinates = VK_FALSE;

    samp.upload();

    defaultSampler = addSampler(std::move(samp));
}

void ResourceManager::destroy()
{
    for (auto& it : images) it.destroy();
    for (auto& it : buffers) it.destroy();
    for (auto& it : samplers) it.destroy();

    images.clear();
    buffers.clear();
    samplers.clear();
}

ImageResourceHandler ResourceManager::addImage(ImageResource image)
{
    images.push_back(std::move(image));
    return images.size() - 1;
}

ImageResource& ResourceManager::get(const ImageResourceHandler& handler)
{
    assert(handler.index);
    assert(*handler.index < images.size());
    return images[*handler.index];
}

BufferResourceHandler ResourceManager::addBuffer(BufferResource buffer)
{
    buffers.push_back(std::move(buffer));
    return buffers.size() - 1;
}

BufferResource& ResourceManager::get(const BufferResourceHandler& handler)
{
    assert(handler.index);
    assert(*handler.index < buffers.size());
    return buffers[*handler.index];
}

SamplerResourceHandler ResourceManager::addSampler(SamplerResource sampler)
{
    samplers.push_back(std::move(sampler));
    return samplers.size() - 1;
}

SamplerResource& ResourceManager::get(const SamplerResourceHandler& handler)
{
    assert(handler.index);
    assert(*handler.index < samplers.size());
    return samplers[*handler.index];
}
