#include "texture.h"

vk::ImageView Texture::view() const
{
    return imageHandler->image.view;
}

vk::Sampler Texture::sampler() const
{
    return samplerHandler->sampler;
}

void Texture::prepare()
{
    if (dummy()) return;
    imageHandler->upload();
    samplerHandler->upload();
}

bool Texture::ready() const
{
    return  gpuIndex == -1 ||
            (imageHandler->mLocation & ResourceLocation::UploadedBit &&
            samplerHandler->mLocation & ResourceLocation::UploadedBit &&
            gpuIndex >= 0);
}

bool Texture::valid() const
{
    return gpuIndex == -1 || (imageHandler.valid() && samplerHandler.valid());
}

bool Texture::dummy() const
{
    return gpuIndex == -1;
}

bool Texture::operator==(const Texture& that) const
{
    return imageHandler == that.imageHandler && samplerHandler == that.samplerHandler;
}

bool Texture::operator!=(const Texture& that) const
{
    return !(*this == that);
}

Texture Texture::empty(std::string name)
{
    Texture th;
    th.mName = std::move(name);
    return th;
}

Texture Texture::dummy(std::string name)
{
    Texture th;
    th.mName = std::move(name);
    th.gpuIndex = -1;
    return th;
}

std::string Texture::typeName() {
    return "Texture";
}
