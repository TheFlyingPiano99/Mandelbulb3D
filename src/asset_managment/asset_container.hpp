#ifndef VULKAN_INTRO_ASSET_CONTAINER_HPP
#define VULKAN_INTRO_ASSET_CONTAINER_HPP

#include "asset_base.hpp"
#include "constants.h"

template <typename Asset>
class AssetContainer {
public:
    AssetID add(Asset asset)
    {
        std::unique_lock<std::recursive_mutex> lock{mMutex};
        auto key = asset.name_str();

        auto _l1 = logger.PushContext("Adding asset of type ", Asset::typeName(), " with name: ", key);

        auto found = idMap.find(key);
        int dupId = 0; bool collision = false;
        while (found != idMap.end()) {
            collision = true;
            key = asset.name_str() + "_" + std::to_string(++dupId);
            found = idMap.find(key);
        }

        if (collision) {
            if (cLogAssetLoading) logger.LogInfo("Name duplication detected. Using ", key, " instead of ", asset.name(), ".");
            asset.setName(key);
        }

        assets.push_back(std::make_unique<Asset>(std::move(asset)));
        auto id = (AssetID)(assets.size() - 1);
        idMap[key] = id;
        return id;
    }

    AssetID translate(const std::string& name)
    {
        std::unique_lock<std::recursive_mutex> lock{mMutex};
        auto found = idMap.find(name);
        if (found == idMap.end())
        {
            logger.LogError("Requested asset: ", name, " of type ", Asset::typeName(), " is not found. Replacing with default asset.");
            assets.push_back(std::make_unique<Asset>(std::move(Asset::empty(name))));
            auto id = AssetID(assets.size() - 1);
            idMap[name] = id;
            return id;
        }
        return found->second;
    }

    AssetID translateIfExists(const std::string& name)
    {
        std::unique_lock<std::recursive_mutex> lock{mMutex};
        auto found = idMap.find(name);
        if (found == idMap.end()) return invalidAssetID();
        return found->second;
    }

    std::vector<std::string_view> getAvailableAssetNames()
    {
        std::unique_lock<std::recursive_mutex> lock{ mMutex };
        std::vector<std::string_view> ret;
        for (auto& it : idMap) {
            ret.push_back(it.first);
        }
        return ret;
    }

    auto size() const 
    {
        std::unique_lock<std::recursive_mutex> lock{mMutex};
        return assets.size();
    }

    Asset& operator[](AssetID id) 
    {
        std::unique_lock<std::recursive_mutex> lock{mMutex};
        assert(id < assets.size()); 
        return *assets[id];
    }

public:
    AssetHandler<Asset> defaultAsset;

protected:
    std::unordered_map<std::string, AssetID> idMap;
    std::vector<std::unique_ptr<Asset>> assets;
    mutable std::recursive_mutex mMutex;
};

#endif//VULKAN_INTRO_ASSET_CONTAINER_HPP
