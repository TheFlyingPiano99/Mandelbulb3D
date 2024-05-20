#ifndef VULKAN_INTRO_ASSET_MANAGER_H
#define VULKAN_INTRO_ASSET_MANAGER_H

#include "asset_container.hpp"
#include "texture.h"
#include "mesh.h"
#include "material.h"
#include "coloured_mesh.h"
#include "composite_mesh.h"

namespace std {
    template <> struct hash<std::pair<unsigned int, vk::Sampler>>
    {
        size_t operator()(const std::pair<unsigned int, vk::Sampler>& x) const
        {
            return x.first ^ (unsigned long long)(VkSampler)x.second ^ 0x9e3779b9;
        }
    };
}

class AssetManager {
public:
    static AssetManager& Instance() { static AssetManager instance; return instance; }

    AssetHandler<Texture> add(Texture texture);
    AssetHandler<Material> add(Material material);
    AssetHandler<Mesh> add(Mesh mesh);
    AssetHandler<ColouredMesh> add(ColouredMesh colouredMesh);
    AssetHandler<CompositeMesh> add(CompositeMesh compositeMesh);

    AssetHandler<Texture> getTexture(const std::string& name);
    AssetHandler<Mesh> getMesh(const std::string& name);
    AssetHandler<Material> getMaterial(const std::string& name);
    AssetHandler<ColouredMesh> getColouredMesh(const std::string& name);
    AssetHandler<CompositeMesh> getCompositeMesh(const std::string& name);
    void loadCompositeMesh(const std::string& name);

    std::vector<std::string_view> getAvailableCompositeMeshNames();

    Texture& get(const TextureHandler& handler);
    Mesh& get(const MeshHandler& handler);
    Material& get(const MaterialHandler& handler);
    ColouredMesh& get(const ColouredMeshHandler& handler);
    CompositeMesh& get(const CompositeMeshHandler& handler);

    template <typename Asset>
    Asset& getDefault();

    void initialize();
    void destroy();

private:
    AssetManager() = default;

    void initializeDefaultAssets();
    void initializeBuffers();
    void initializeDescriptors();

public:
    vk::DescriptorSet materialDescriptorSet;
    VulkanBuffer indexValueBuffer;
    vk::DescriptorSetLayout materialSetLayout;
    TextureHandler dummyTexture;

private:
    AssetContainer<Texture> textureManager;
    AssetContainer<Mesh> meshManager;
    AssetContainer<Material> materialManager;
    AssetContainer<ColouredMesh> colouredMeshManager;
    AssetContainer<CompositeMesh> compositeMeshManager;

    uint32_t actualMaxSampledTextureIndex = 0;
    uint32_t actualMaxMaterialEntryIndex = 0;

    const uint32_t MAX_MATERIAL_ENTRY_COUNT = 1000, MAX_TEXTURE_COUNT = 100;

    vk::DescriptorPool materialDescriptorPool;
};

inline auto& theAssetManager = AssetManager::Instance();

#endif//VULKAN_INTRO_ASSET_MANAGER_H
