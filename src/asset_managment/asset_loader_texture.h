#ifndef VULKAN_INTRO_ASSET_LOADER_TEXTURE_H
#define VULKAN_INTRO_ASSET_LOADER_TEXTURE_H

#include "texture.h"

class AssetManager;

TextureHandler handleTexture(AssetManager& am, const std::string& texName, const std::string& colorSpace, bool sRGB, bool mipMapped);
TextureHandler handleTextureCombination(AssetManager& am, const std::vector<std::pair<std::string, std::string>>& textures, bool mipMapped);

#endif//VULKAN_INTRO_ASSET_LOADER_TEXTURE_H
