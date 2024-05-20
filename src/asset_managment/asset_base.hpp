#ifndef VULKAN_INTRO_ASSET_BASE_HPP
#define VULKAN_INTRO_ASSET_BASE_HPP

#include "asset_id.h"

class AssetManager;

class AssetBase {
public:
    std::string_view name() const { return mName; }
    const std::string& name_str() const { return mName; }
    void setName(std::string name) { mName = std::move(name); }

protected:
    std::string mName;
};

template <typename Asset>
class AssetHandler {
public:
    AssetHandler() : id{invalidAssetID()} {}
    AssetHandler(AssetID id) : id{id} {}
    Asset* operator->();
    const Asset* operator->() const;
    bool valid() const { return !isInvalid(id); }
    bool operator==(const AssetHandler& that) const { return id == that.id; }

    friend class AssetManager;

private:
    AssetID id;
};


#endif//VULKAN_INTRO_ASSET_BASE_HPP
