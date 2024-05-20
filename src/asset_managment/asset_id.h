#ifndef VULKAN_INTRO_ASSET_ID_H
#define VULKAN_INTRO_ASSET_ID_H

using AssetID = unsigned int;

inline constexpr AssetID invalidAssetID()
{
    return std::numeric_limits<AssetID>::max();
}

inline constexpr bool isInvalid(AssetID id)
{
    return id == invalidAssetID();
}

#endif//VULKAN_INTRO_ASSET_ID_H
