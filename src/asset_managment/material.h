#ifndef VULKAN_INTRO_MATERIAL_H
#define VULKAN_INTRO_MATERIAL_H

#include "texture.h"

struct Material : public AssetBase {
    static Material empty(std::string name);
    static std::string typeName();

    void prepare();
    bool ready() const;
    bool valid() const;

    void setGPULocationIndex(int32_t index);

    bool operator==(const Material& that) const;
    bool operator!=(const Material& that) const;

    std::vector<TextureHandler> textures;
    std::vector<float> values;
    bool hasTransparency = false;

    int32_t gpuIndex = -1;

private:
    Material() = default;
};

using MaterialHandler = AssetHandler<Material>;

template<> Material* AssetHandler<Material>::operator->();

#endif//VULKAN_INTRO_MATERIAL_H
