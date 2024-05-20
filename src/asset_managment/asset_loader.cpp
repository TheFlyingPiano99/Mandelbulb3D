#include "asset_loader.h"
#include "asset_manager.h"

namespace fs = std::filesystem;

AssetLoader& AssetLoader::instance()
{
    static AssetLoader instance{ AssetManager::Instance() };
    return instance;
}

void AssetLoader::addRootAssetLocationDirectory(std::string directory)
{
    rootPaths.push_back(std::move(directory));
}

void AssetLoader::collectFiles()
{
    for (auto& root : rootPaths) {
        for (auto& p : fs::recursive_directory_iterator(root)) {
            if (std::string ext = p.path().extension().string(); p.is_regular_file() && loaders.contains(ext.substr(1))) {
                auto name = p.path().lexically_relative(root).generic_string();
                auto cmh = CompositeMesh::empty(name);
                cmh.path = p.path().relative_path().generic_string();
                am.add(std::move(cmh));
            }
        }
    }
}

void AssetLoader::validateAssets(std::vector<std::string> names, CompatibilityDescriptor compatibilityDescriptor)
{
    scheduledAssets = scheduledAssets + names.size();
    inProgress = true; 
    for (size_t i = 0; i < names.size(); completedAssets = completedAssets + 1, ++i) {
        auto& name = names[i];
        auto _l1 = logger.PushContext("Validating ", CompositeMesh::typeName(), " - ", name, ":");

        auto cmh = am.getCompositeMesh(name);
        auto& mesh = am.get(cmh);

        if (mesh.path.empty()) {
            continue;
        }

        if (mesh.valid()) {
            mesh.prepare();
            continue;
        }

        auto ext = mesh.path.substr(mesh.path.rfind('.') + 1);
        auto found = loaders.find(ext);
        if (found == loaders.end()) {
            logger.LogError(ext, " not recognized from ", mesh.path);
            continue;
        } else {
            (this->*found->second)(mesh, compatibilityDescriptor);
        }
    }
    scheduledAssets = scheduledAssets - names.size();
    completedAssets = completedAssets - names.size();
}

void AssetLoader::initialize()
{
    loaders["obj"] = &AssetLoader::validateObjFile;
    loaders["glb"] = &AssetLoader::validateGlbFile;
    loaders["gltf"] = &AssetLoader::validateGltfFile;
}

void AssetLoader::destroy()
{
}
