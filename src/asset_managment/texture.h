#ifndef VULKAN_INTRO_TEXTURE_HANDLER_H
#define VULKAN_INTRO_TEXTURE_HANDLER_H

#include "asset_base.hpp"
#include "resources/resource_manager.hpp"

struct Texture : public AssetBase {
    static Texture empty(std::string name);
    static Texture dummy(std::string name);
    static std::string typeName();

    void prepare();

    bool valid() const;
    bool ready() const;
    bool dummy() const;

    bool operator==(const Texture& that) const;
    bool operator!=(const Texture& that) const;

    vk::ImageView view() const;
    vk::Sampler sampler() const;

    ImageResourceHandler imageHandler;
    SamplerResourceHandler samplerHandler;
    bool hasTransparency = false;

    int32_t gpuIndex = -2;

private:
    Texture() = default;
};

using TextureHandler = AssetHandler<Texture>;

template<> Texture* AssetHandler<Texture>::operator->();

#endif//VULKAN_INTRO_TEXTURE_HANDLER_H
