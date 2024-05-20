#ifndef VULKAN_INTRO_COMPATIBILITY_DESCRIPTOR_H
#define VULKAN_INTRO_COMPATIBILITY_DESCRIPTOR_H

#include "material_traits.h"

struct CompatibilityDescriptor {
    MaterialDescriptor materialDescriptor;

    bool operator==(const CompatibilityDescriptor& that) const { return materialDescriptor == that.materialDescriptor; }
    bool operator!=(const CompatibilityDescriptor& that) const { return !(*this == that); }
};

inline auto theCompatibilityDescriptor = CompatibilityDescriptor{
    {
        { 
            {MaterialElemTypeTexture::DiffuseTexture, true, true},
            {MaterialElemTypeTexture::NormalTexture, true, false},
            {MaterialElemTypeTexture::CombinedMRO, true, false}
        },
        {
            MaterialElemTypeValue::DiffuseValueR,
            MaterialElemTypeValue::DiffuseValueG,
            MaterialElemTypeValue::DiffuseValueB,
            MaterialElemTypeValue::OpacityValueR,
            MaterialElemTypeValue::MetallicValueR,
            MaterialElemTypeValue::RoughnessValueR
        }
    }
};

#endif//VULKAN_INTRO_COMPATIBILITY_DESCRIPTOR_H
