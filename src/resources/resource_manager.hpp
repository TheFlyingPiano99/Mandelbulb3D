#ifndef VULKAN_INTRO_RESOURCE_MANAGER_HPP
#define VULKAN_INTRO_RESOURCE_MANAGER_HPP

#include "gpu_resources.h"

class ResourceManager;

template <typename Resource>
class ResourceHandler {
public:
    ResourceHandler() : index{std::nullopt} {}
    ResourceHandler(uint32_t idx) : index(idx) {}
    ResourceHandler(uint64_t idx) : index((uint32_t)idx) {}
    Resource* operator->();
    const Resource* operator->() const;
    bool valid() const { return !!index; }
    bool operator==(const ResourceHandler& that) const { return index == that.index; }
    bool operator!=(const ResourceHandler& that) const { return index != that.index; }

    friend class ResourceManager;
private:
    std::optional<uint32_t> index;
};

using BufferResourceHandler = ResourceHandler<BufferResource>;
using ImageResourceHandler = ResourceHandler<ImageResource>;
using SamplerResourceHandler = ResourceHandler<SamplerResource>;

class ResourceManager {
public:
    static ResourceManager& instance() { static ResourceManager instance; return instance; }

    void initialize();
    void destroy();

    ResourceHandler<ImageResource> addImage(ImageResource image);
    ImageResource& get(const ResourceHandler<ImageResource>& handler);

    ResourceHandler<BufferResource> addBuffer(BufferResource buffer);
    BufferResource& get(const ResourceHandler<BufferResource>& handler);

    ResourceHandler<SamplerResource> addSampler(SamplerResource buffer);
    SamplerResource& get(const ResourceHandler<SamplerResource>& handler);

public:
    SamplerResourceHandler defaultSampler;

private:
    ResourceManager() = default;

private:
    std::vector<ImageResource> images;
    std::vector<BufferResource> buffers;
    std::vector<SamplerResource> samplers;
};

inline auto& theResourceManager = ResourceManager::instance();

template <typename Resource>
Resource* ResourceHandler<Resource>::operator->()
{
    return &theResourceManager.get(*this);
}

template <typename Resource>
const Resource* ResourceHandler<Resource>::operator->() const
{
    return &theResourceManager.get(*this);
}

#endif//VULKAN_INTRO_RESOURCE_MANAGER_HPP
