#include "material.h"

void Material::prepare()
{
    for (auto& te : textures) te->prepare();
}

bool Material::ready() const
{
    for (auto& te : textures) 
        if (!te->ready()) return false;
    return gpuIndex >= 0;
}

bool Material::operator==(const Material& that) const
{
    return textures == that.textures && values == that.values;
}

bool Material::operator!=(const Material& that) const
{
    return !(*this == that);
}

Material Material::empty(std::string name)
{
    Material mh;
    mh.mName = std::move(name);
    mh.gpuIndex = -1;
    return mh;
}

bool Material::valid() const
{
    return !textures.empty() || !values.empty();
}

void Material::setGPULocationIndex(int32_t index)
{
    gpuIndex = index;
}

std::string Material::typeName() {
    return "Material";
}
