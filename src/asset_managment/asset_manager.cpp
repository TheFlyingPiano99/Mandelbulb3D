#include "asset_manager.h"
#include "compatibility_descriptor.h"
#include "vulkan_layer.h"
#include "rendering/renderer_dependency_provider.h"
#include "asset_loader.h"
#include <tinyobjloader/tiny_obj_loader.h>

bool readFromTinyObjLoaderThings(Mesh& mesh, const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape);

TextureHandler AssetManager::add(Texture texture)
{
    auto th = TextureHandler{textureManager.add(std::move(texture))};
    if (!th->valid()) return th;

    th->prepare();
    if (th->ready()) return th; // Ha redirect elve lett default-ra, akkor nem kell feltolteni

    vk::DescriptorImageInfo info;
    info.imageView = th->view();
    info.sampler = th->sampler();
    info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::WriteDescriptorSet descriptorWrite;
    descriptorWrite.dstSet = materialDescriptorSet;
    descriptorWrite.dstBinding = 1;
    descriptorWrite.dstArrayElement = actualMaxSampledTextureIndex;
    descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &info;

    th->gpuIndex = (int32_t)actualMaxSampledTextureIndex++;

    assert(actualMaxSampledTextureIndex <= MAX_TEXTURE_COUNT && "Reached maximum number of textures!");

    theVulkanLayer.device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);

    return th;
}

MaterialHandler AssetManager::add(Material material)
{
    auto& vl = theVulkanLayer;
    auto mh = MaterialHandler{materialManager.add(std::move(material))};
    if (!mh->valid()) return mh;

    mh->prepare();
    if (mh->ready()) return mh; // Ha redirect elve lett default-ra, akkor nem kell feltolteni
    
    if (mh->textures.size() + mh->values.size() != 0) {
        auto cmd = vl.beginSingleTimeTransferCommand();
        std::vector<float> data;
        data.reserve(mh->textures.size() + mh->values.size());
        for (auto& te : mh->textures) {
            data.push_back(te->gpuIndex + 0.5f);
        }
        for (auto v : mh->values) {
            data.push_back(v);
        }
        cmd.updateBuffer(indexValueBuffer.buffer, actualMaxMaterialEntryIndex * sizeof(float), data.size() * sizeof(float), data.data());
        vl.endSingleTimeTransferCommand(cmd);

        mh->setGPULocationIndex((int32_t)actualMaxMaterialEntryIndex);

        actualMaxMaterialEntryIndex += mh->textures.size() + mh->values.size();

        assert(actualMaxMaterialEntryIndex <= MAX_MATERIAL_ENTRY_COUNT);
    } else {
        mh->setGPULocationIndex(-1);
    }

    return mh;
}

MeshHandler AssetManager::add(Mesh mesh)
{
    return MeshHandler{meshManager.add(std::move(mesh))};
}

ColouredMeshHandler AssetManager::add(ColouredMesh colouredMesh)
{
    return ColouredMeshHandler{colouredMeshManager.add(std::move(colouredMesh))};
}

CompositeMeshHandler AssetManager::add(CompositeMesh compositeMesh)
{
    return CompositeMeshHandler{compositeMeshManager.add(std::move(compositeMesh))};
}

TextureHandler AssetManager::getTexture(const std::string& name)
{
    return TextureHandler{textureManager.translate(name)};
}

MeshHandler AssetManager::getMesh(const std::string& name)
{
    return MeshHandler{meshManager.translate(name)};
}

MaterialHandler AssetManager::getMaterial(const std::string& name)
{
    return MaterialHandler{materialManager.translate(name)};
}

ColouredMeshHandler AssetManager::getColouredMesh(const std::string& name)
{
    return ColouredMeshHandler{colouredMeshManager.translate(name)};
}

CompositeMeshHandler AssetManager::getCompositeMesh(const std::string& name)
{
    return CompositeMeshHandler{compositeMeshManager.translate(name)};
}

void AssetManager::loadCompositeMesh(const std::string& name)
{
    theAssetLoader.validateAssets({ name }, theCompatibilityDescriptor);
}

std::vector<std::string_view> AssetManager::getAvailableCompositeMeshNames()
{
    return compositeMeshManager.getAvailableAssetNames();
}

void AssetManager::initialize()
{
    initializeBuffers();
    initializeDescriptors();
    initializeDefaultAssets();
}

void AssetManager::destroy()
{
    auto& vl = theVulkanLayer;

    for (uint32_t i = 0; i < compositeMeshManager.size(); ++i) {
        compositeMeshManager[i].destroy();
    }

    vl.device.destroy(materialDescriptorPool);
    vl.device.destroy(materialSetLayout);
    indexValueBuffer.destroy();
}

void AssetManager::initializeDefaultAssets()
{
    std::string box_obj = R"(
        v  0.0  0.0  0.0
        v  0.0  0.0  1.0
        v  0.0  1.0  0.0
        v  0.0  1.0  1.0
        v  1.0  0.0  0.0
        v  1.0  0.0  1.0
        v  1.0  1.0  0.0
        v  1.0  1.0  1.0

        vt  0.0  0.0
        vt  0.0  0.0
        vt  0.0  1.0
        vt  0.0 -1.0
        vt  1.0  0.0
        vt -1.0  0.0

        vn  0.0  0.0  1.0
        vn  0.0  0.0 -1.0
        vn  0.0  1.0  0.0
        vn  0.0 -1.0  0.0
        vn  1.0  0.0  0.0
        vn -1.0  0.0  0.0
         
        f  1//2  7//2  5//2
        f  1//2  3//2  7//2 
        f  1//6  4//6  3//6 
        f  1//6  2//6  4//6 
        f  3//3  8//3  7//3 
        f  3//3  4//3  8//3 
        f  5//5  7//5  8//5 
        f  5//5  8//5  6//5 
        f  1//4  5//4  6//4 
        f  1//4  6//4  2//4 
        f  2//1  6//1  8//1 
        f  2//1  8//1  4//1  
    )";
    std::istringstream is{ box_obj };
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, &is);

    auto mh = Mesh::empty("__defaultCubeMesh");

    mh.topology = vk::PrimitiveTopology::eTriangleList;
    mh.parentTranslation = { 0, 0, 0 };

    bool loaded [[maybe_unused]] = readFromTinyObjLoaderThings(mh, attrib, shapes[0]);
    assert(loaded);

    mh.prepare();
    meshManager.defaultAsset = add(std::move(mh));

    ImageResource img;
    img.width = 64;
    img.height = 64;
    img.channel = 4;
    img.componentSize = 1;
    img.sRGB = false;
    img.mipLevel = 1;   
    img.mLocation = ResourceLocation::InMemoryBit;

    img.data.resize(img.width * img.height * 4);
    
    for (uint32_t x = 0; x < img.width; ++x) {
        for (uint32_t y = 0; y < img.height; ++y) {
            std::byte* act = img.data.data() + y * img.width * img.channel + x * img.channel;
            act[0] = act[1] = (std::byte)0;
            act[3] = (std::byte)255;
            int s = x / 8;
            if (s % 2 == 0) act[2] = (std::byte)255;
            else act[2] = (std::byte)127;
        }
    }
    img.pixelFormat = vk::Format::eR8G8B8A8Unorm;
    img.transparent = false;
    img.usage = vk::ImageUsageFlagBits::eSampled;

    img.upload();
    auto ih = theResourceManager.addImage(std::move(img));

    auto th = Texture::empty("__defaultTexture");
    th.imageHandler = ih;
    th.samplerHandler = theResourceManager.defaultSampler;
    th.hasTransparency = false;
    th.prepare();
    textureManager.defaultAsset = add(th);

    dummyTexture = textureManager.add(Texture::dummy("__DummyTexture"));

    auto mth = Material::empty("__defaultMaterial");
    mth.textures.resize(theCompatibilityDescriptor.materialDescriptor.textureDescriptors.size(), dummyTexture);
    mth.values.resize(theCompatibilityDescriptor.materialDescriptor.valueDescriptors.size(), 0.5f);
    mth.hasTransparency = false;
    mth.prepare();
    materialManager.defaultAsset = add(std::move(mth));

    auto cmh = ColouredMesh::empty("__defaultColouredMesh");
    cmh.mesh = meshManager.defaultAsset;
    cmh.material = materialManager.defaultAsset;
    cmh.rendererDependencyIndex = theRendererDependencyProvider.addDependency(RendererDependency{ cmh.mesh->layout, cmh.material->hasTransparency });

    colouredMeshManager.defaultAsset = add(std::move(cmh));

    auto cmph = CompositeMesh::empty("__defaultCompositeMesh");
    cmph.meshHandlers.push_back(CompositeMeshData{glm::translate(glm::vec3{0, 0, 0}), colouredMeshManager.defaultAsset});
    compositeMeshManager.defaultAsset = add(std::move(cmph));
}

void AssetManager::initializeDescriptors()
{
    auto& vl = theVulkanLayer;

    vk::DescriptorSetLayoutBinding indexBinding = {};
    indexBinding.binding = 0;
    indexBinding.descriptorCount = 1;
    indexBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    indexBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding textureBinding = {};
    textureBinding.binding = 1;
    textureBinding.descriptorCount = MAX_TEXTURE_COUNT;
    textureBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    textureBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    textureBinding.pImmutableSamplers = nullptr;

    static std::array<vk::DescriptorSetLayoutBinding, 2> materialBindings = {
            indexBinding, textureBinding
    };

    static std::array<vk::DescriptorBindingFlagsEXT, 2> materialBindingFlags = {
            vk::DescriptorBindingFlagsEXT{}, vk::DescriptorBindingFlagBitsEXT::ePartiallyBound | vk::DescriptorBindingFlagBitsEXT::eUpdateUnusedWhilePending | vk::DescriptorBindingFlagBitsEXT::eUpdateAfterBind
    };

    vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT extInfo = {};
    extInfo.pBindingFlags = materialBindingFlags.data();
    extInfo.bindingCount = materialBindingFlags.size();

    vk::DescriptorSetLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.pBindings = materialBindings.data();
    layoutCreateInfo.bindingCount = materialBindings.size();
    layoutCreateInfo.flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
    layoutCreateInfo.pNext = &extInfo;

    materialSetLayout = vl.device.createDescriptorSetLayout(layoutCreateInfo);

    static std::array<vk::DescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[0].descriptorCount = MAX_TEXTURE_COUNT;
    poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
    poolSizes[1].descriptorCount = 1;

    vk::DescriptorPoolCreateInfo poolInfo = {};
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;

    materialDescriptorPool = vl.device.createDescriptorPool(poolInfo);

    vk::DescriptorSetAllocateInfo allocateInfo = {};
    allocateInfo.descriptorPool = materialDescriptorPool;
    allocateInfo.pSetLayouts = &materialSetLayout;
    allocateInfo.descriptorSetCount = 1;

    materialDescriptorSet = vl.device.allocateDescriptorSets(allocateInfo)[0];

    vk::DescriptorBufferInfo indexInfo = {};
    indexInfo.buffer = indexValueBuffer.buffer;
    indexInfo.offset = 0;
    indexInfo.range = VK_WHOLE_SIZE;

    std::array<vk::WriteDescriptorSet, 1> writeDescriptorSets = {};
    writeDescriptorSets[0].dstSet = materialDescriptorSet;
    writeDescriptorSets[0].dstBinding = 0;
    writeDescriptorSets[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    writeDescriptorSets[0].descriptorCount = 1;
    writeDescriptorSets[0].pBufferInfo = &indexInfo;


    vl.device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);
}

void AssetManager::initializeBuffers()
{
    vk::DeviceSize bufferSize = MAX_MATERIAL_ENTRY_COUNT * sizeof(float);
    indexValueBuffer = theVulkanLayer.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
}

Texture& AssetManager::get(const TextureHandler& handler) { return textureManager[handler.id]; }
Mesh& AssetManager::get(const MeshHandler& handler) { return meshManager[handler.id]; }
Material& AssetManager::get(const MaterialHandler& handler) { return materialManager[handler.id]; }
ColouredMesh& AssetManager::get(const ColouredMeshHandler& handler) { return colouredMeshManager[handler.id]; }
CompositeMesh& AssetManager::get(const CompositeMeshHandler& handler) { return compositeMeshManager[handler.id]; }

template <> Texture& AssetManager::getDefault<>() { return textureManager[textureManager.defaultAsset.id]; }
template <> Mesh& AssetManager::getDefault<>() { return meshManager[meshManager.defaultAsset.id]; }
template <> Material& AssetManager::getDefault<>() { return materialManager[materialManager.defaultAsset.id]; }
template <> ColouredMesh& AssetManager::getDefault<>() { return colouredMeshManager[colouredMeshManager.defaultAsset.id]; }
template <> CompositeMesh& AssetManager::getDefault<>() { return compositeMeshManager[compositeMeshManager.defaultAsset.id]; }

template<> Texture* AssetHandler<Texture>::operator->()
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<Texture>();
}

template<> const Texture* AssetHandler<Texture>::operator->() const
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<Texture>();
}

template<> Mesh* AssetHandler<Mesh>::operator->()
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<Mesh>();
}

template<> const Mesh* AssetHandler<Mesh>::operator->() const
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<Mesh>();
}

template<> Material* AssetHandler<Material>::operator->()
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<Material>();
}

template<> const Material* AssetHandler<Material>::operator->() const
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<Material>();
}

template<> ColouredMesh* AssetHandler<ColouredMesh>::operator->()
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<ColouredMesh>();
}

template<> const ColouredMesh* AssetHandler<ColouredMesh>::operator->() const
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<ColouredMesh>();
}

template<> CompositeMesh* AssetHandler<CompositeMesh>::operator->()
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<CompositeMesh>();
}

template<> const CompositeMesh* AssetHandler<CompositeMesh>::operator->() const
{
    if (auto& a = theAssetManager.get(*this); a.valid()) return &a;
    else return &theAssetManager.getDefault<CompositeMesh>();
}
