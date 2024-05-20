#ifndef VULKAN_INTRO_MATERIAL_TRAITS_H
#define VULKAN_INTRO_MATERIAL_TRAITS_H

enum class MaterialElemTypeTexture {
    DiffuseTexture,
    NormalTexture,
    CombinedMRO
};

enum class MaterialElemTypeValue {
    DiffuseValueR, DiffuseValueG, DiffuseValueB,
    OpacityValueR,
    RoughnessValueR,
    MetallicValueR,
    NormalScaleR
};

struct TextureDescriptor {
    MaterialElemTypeTexture type;
    bool mipMapped, sRGB;

    bool operator==(const TextureDescriptor& that) const { return type == that.type; }
    bool operator!=(const TextureDescriptor& that) const { return !(*this == that); }
};

struct MaterialDescriptor {
    std::vector<TextureDescriptor> textureDescriptors;
    std::vector<MaterialElemTypeValue> valueDescriptors;

    bool operator==(const MaterialDescriptor& that) const { return textureDescriptors == that.textureDescriptors && valueDescriptors == that.valueDescriptors; }
    bool operator!=(const MaterialDescriptor& that) const { return !(*this == that); }
};

namespace std {
    template <> struct hash<vk::SamplerCreateInfo>
    {
        size_t operator()(const vk::SamplerCreateInfo& x) const;
    };
}

#endif//VULKAN_INTRO_MATERIAL_TRAITS_H
