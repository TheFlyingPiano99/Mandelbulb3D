#ifndef VULKAN_INTRO_ASSET_LOADER_H
#define VULKAN_INTRO_ASSET_LOADER_H

#include "compatibility_descriptor.h"

class AssetManager;
struct CompositeMesh;

namespace tinygltf {
    class Model;
}

class AssetLoader {
public:
    explicit AssetLoader(AssetManager& assetManager) : am{assetManager} {}
    static AssetLoader& instance();

    void initialize();
    void destroy();
    void addRootAssetLocationDirectory(std::string directory);
    void collectFiles();
    void validateAssets(std::vector<std::string> names, CompatibilityDescriptor compatibilityDescriptor);

private:
    using LoaderFunction = void (AssetLoader::*)(CompositeMesh& cmh, CompatibilityDescriptor compat);
    void validateObjFile(CompositeMesh& cmh, CompatibilityDescriptor compat);
    void validateGlbFile(CompositeMesh& cmh, CompatibilityDescriptor compat);
    void validateGltfFile(CompositeMesh& cmh, CompatibilityDescriptor compat);

    void loadGltfData(CompositeMesh& cmh, CompatibilityDescriptor compat, tinygltf::Model& model);

private:
    AssetManager& am;
    std::vector<std::string> rootPaths;
    std::map<std::string, LoaderFunction> loaders;
    std::atomic<size_t> scheduledAssets = 0;
    std::atomic<size_t> completedAssets = 0;
    std::atomic<bool> inProgress = false;
};

inline auto& theAssetLoader = AssetLoader::instance();

#endif//VULKAN_INTRO_ASSET_LOADER_H
