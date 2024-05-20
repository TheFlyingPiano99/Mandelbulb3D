#include "asset_loader.h"
#include "asset_manager.h"
#include "asset_loader_mesh.hpp"
#include "rendering/renderer_dependency_provider.h"
#include "resources/resource_manager.hpp"
#define TINYGLTF_USE_CPP14
#include <tinygltf/tiny_gltf.h>

using namespace tinygltf;

static vk::SamplerAddressMode convertAddressMode(int tinygltfAddressMode)
{
    switch (tinygltfAddressMode) {
    case TINYGLTF_TEXTURE_WRAP_REPEAT:
        return vk::SamplerAddressMode::eRepeat;
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
        return vk::SamplerAddressMode::eClampToEdge;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
        return vk::SamplerAddressMode::eMirroredRepeat;
    default:
        return vk::SamplerAddressMode::eRepeat;
    }
}

static std::tuple<vk::Filter, vk::SamplerMipmapMode> convertFilter(int tinygltfFilter)
{
    switch (tinygltfFilter) {
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
        return {vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest};
    case TINYGLTF_TEXTURE_FILTER_LINEAR:
        return {vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest};
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
        return {vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest};
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
        return {vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest};
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
        return {vk::Filter::eNearest, vk::SamplerMipmapMode::eLinear};
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
        return {vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    default:
        return {vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest};
    }
}

static float getMaxLod(int tingltfMinFilter)
{
    switch (tingltfMinFilter) {
    default:
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_LINEAR:
        return 0.25f;
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
        return 2000.f;
    }
}

template <typename T> using HandlerHolder = std::vector<std::optional<T>>;

struct Holders {
    HandlerHolder<ImageResourceHandler> images;
    HandlerHolder<BufferResourceHandler> buffers;
    HandlerHolder<SamplerResourceHandler> samplers;
    HandlerHolder<TextureHandler> textures;
};

static ImageResourceHandler requestImage(const Model& model, Holders& holders, uint32_t index, bool srgb, bool mipmap)
{
    if (!holders.images[index]) {
        auto& image = model.images[index];

        ImageResource img;
        img.width = (uint32_t)image.width;
        img.height = (uint32_t)image.height;
        img.channel = (uint32_t)image.component;
        img.componentSize = (uint32_t)tinygltf::GetComponentSizeInBytes((uint32_t)image.pixel_type);
        img.sRGB = srgb;
        img.mipLevel = mipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(img.width, img.height)))) + 1 : 1;
        img.mLocation = ResourceLocation::InMemoryBit;
        img.data = std::move((std::vector<std::byte>&)const_cast<std::vector<uint8_t>&>(image.image));
        img.calculatePixelFormat(image.pixel_type);
        img.calculateTransparency();
        img.usage = vk::ImageUsageFlagBits::eSampled;

        img.upload();
        holders.images[index] = theResourceManager.addImage(std::move(img));
    }
    return holders.images[index].value();
}

[[maybe_unused]]
static BufferResourceHandler requestBuffer(const Model& model, Holders& holders, size_t index)
{
    if (!holders.buffers[index]) {
        auto& buffer = model.buffers[index];

        BufferResource buf;
        buf.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;
        buf.data = std::move((std::vector<std::byte>&)const_cast<std::vector<uint8_t>&>(buffer.data));
        buf.mLocation = ResourceLocation::InMemoryBit;

        buf.upload();
        holders.buffers[index] = theResourceManager.addBuffer(std::move(buf));
    }
    return holders.buffers[index].value();
}

static SamplerResourceHandler requestSampler(const Model& model, Holders& holders, int i)
{
    if (i < 0) return theResourceManager.defaultSampler;
    auto index = (size_t)i;
    if (!holders.samplers[index]) {
        auto& sampler = model.samplers[index];

        SamplerResource samp;
        std::tie(samp.info.magFilter, samp.info.mipmapMode) = convertFilter(sampler.magFilter);
        std::tie(samp.info.minFilter, samp.info.mipmapMode) = convertFilter(sampler.minFilter);
        samp.info.addressModeU = convertAddressMode(sampler.wrapS);
        samp.info.addressModeV = convertAddressMode(sampler.wrapT);
        samp.info.addressModeW = convertAddressMode(sampler.wrapT);

        samp.info.mipLodBias = 0.f;
        samp.info.anisotropyEnable = VK_TRUE;
        samp.info.maxAnisotropy = 16.f;
        samp.info.compareEnable = VK_FALSE;
        samp.info.minLod = 0.f;
        samp.info.maxLod = getMaxLod(sampler.minFilter);
        samp.info.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samp.info.unnormalizedCoordinates = VK_FALSE;

        samp.upload();
        holders.samplers[index] = theResourceManager.addSampler(std::move(samp));
    }
    return holders.samplers[index].value();
}

static TextureHandler requestTexture(AssetManager& am, const std::string& baseName, const Model& model, Holders& holders, uint32_t index, bool srgb, bool mipmap)
{
    if (!holders.textures[index]) {
        auto& texture = model.textures[index];

        auto textureName = baseName + "/textures/";
        textureName += texture.name.empty() ? std::to_string(index) : texture.name;
        auto _l3 = logger.PushContext("Loading texture: ", textureName);

        auto th = ::Texture::empty(textureName);
        th.samplerHandler = requestSampler(model, holders, texture.sampler);
        th.imageHandler = requestImage(model, holders, (uint32_t)texture.source, srgb, mipmap);
        th.hasTransparency = th.imageHandler->transparent;

        holders.textures[index] = am.add(std::move(th));
    }
    return holders.textures[index].value();
}

void AssetLoader::loadGltfData(CompositeMesh& cmh, CompatibilityDescriptor compat, Model& model)
{
    std::string name = (std::string)cmh.name();

    Holders holders;
    holders.images.resize(model.images.size(), std::nullopt);
    holders.buffers.resize(model.buffers.size(), std::nullopt);
    holders.samplers.resize(model.samplers.size(), std::nullopt);
    holders.textures.resize(model.textures.size(), std::nullopt);


    std::vector<MaterialHandler> materials;
    for (int i = 0; auto& material : model.materials) {
        auto materialName = name + "/materials/";
        materialName += material.name.empty() ? std::to_string(i) : material.name;
        auto _l3 = logger.PushContext("Loading material: ", materialName);

        auto mh = ::Material::empty(materialName);

        for (auto& desc : compat.materialDescriptor.textureDescriptors) {
            long long index = -1;
            bool srgb = false, mipmap = true;
            switch (desc.type) { // TODO: support different texture coordinates
                case MaterialElemTypeTexture::DiffuseTexture:
                    index = material.pbrMetallicRoughness.baseColorTexture.index;
                    srgb = true;
                    break;
                case MaterialElemTypeTexture::NormalTexture:
                    index = material.normalTexture.index;
                    break;
                case MaterialElemTypeTexture::CombinedMRO:
                    index = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
                    break;
                default:
                    assert(false && "Unhandled material texture type!");
            }
            if (index >= 0) {
                mh.textures.push_back({requestTexture(am, name, model, holders, (uint32_t)index, srgb, mipmap)});
                if (desc.type == MaterialElemTypeTexture::DiffuseTexture) {
                    mh.hasTransparency |= mh.textures.back()->hasTransparency;
                }
            } else {
                mh.textures.push_back(theAssetManager.dummyTexture);
            }
        }

        for (auto& desc : compat.materialDescriptor.valueDescriptors) {
            double *pValue;
            switch (desc) {
                case MaterialElemTypeValue::DiffuseValueR:
                    pValue = &material.pbrMetallicRoughness.baseColorFactor[0];
                    break;
                case MaterialElemTypeValue::DiffuseValueG:
                    pValue = &material.pbrMetallicRoughness.baseColorFactor[1];
                    break;
                case MaterialElemTypeValue::DiffuseValueB:
                    pValue = &material.pbrMetallicRoughness.baseColorFactor[2];
                    break;
                case MaterialElemTypeValue::OpacityValueR:
                    pValue = &material.pbrMetallicRoughness.baseColorFactor[3];
                    break;
                case MaterialElemTypeValue::RoughnessValueR:
                    pValue = &material.pbrMetallicRoughness.roughnessFactor;
                    break;
                case MaterialElemTypeValue::MetallicValueR:
                    pValue = &material.pbrMetallicRoughness.metallicFactor;
                    break;
                case MaterialElemTypeValue::NormalScaleR:
                    pValue = &material.normalTexture.scale;
                    break;
                default:
                    assert(false && "Unhandled material value type!");
            }

            mh.values.push_back((float)*pValue);
        }

        materials.emplace_back(am.add(std::move(mh)));
        ++i;
    }

    std::vector<glm::mat4> nodeTransforms;
    nodeTransforms.resize(model.nodes.size(), glm::identity<glm::mat4>());

    // Find node parents and calculate transform matrices
    std::vector<int> nodeParents;
    nodeParents.resize(model.nodes.size(), -1);
    for (size_t N = 0; N < model.nodes.size(); ++N) {
        auto& node = model.nodes[N];

        for (auto& c : node.children) {
            nodeParents[(size_t)c] = (int)N;
        }

        glm::vec3 tr = {0, 0, 0}, sc = {1, 1, 1};
        glm::quat rot = glm::identity<glm::quat>();
        glm::mat4 mx = glm::identity<glm::mat4>();

        if (node.matrix.size() == 16) mx = glm::make_mat4x4(node.matrix.data());
        if (node.translation.size() == 3) tr = glm::make_vec3(node.translation.data());
        if (node.scale.size() == 3) sc = glm::make_vec3(node.scale.data());
        if (node.rotation.size() == 4) rot = glm::make_quat(node.rotation.data());

        nodeTransforms[N] = glm::translate(tr) * glm::toMat4(rot) * glm::scale(sc) * mx;
    }

    auto getMatrix = [&](size_t n) -> glm::mat4 {
        glm::mat4 m = nodeTransforms[n];
        while(nodeParents[n] >= 0) {
            n = (size_t)nodeParents[n];
            m = nodeTransforms[n] * m;
        }
        return m;
    };

    for (size_t N = 0; N < model.nodes.size(); ++N) if (model.nodes[N].mesh >= 0) {
        auto& node = model.nodes[N];
        auto& mesh = model.meshes[(size_t)node.mesh];
        auto meshName = name + "/meshes/";
        meshName += mesh.name.empty() ? std::to_string(node.mesh) : mesh.name;
        auto _l3 = logger.PushContext("Loading mesh: ", meshName);

        tinygltf::Skin* skin = (node.skin >= 0) ? &model.skins[(size_t)node.skin] : nullptr;

        for (int j = 0; auto& primitive : mesh.primitives) {
            auto subMeshName = meshName + "/" + std::to_string(j);
            auto _l4 = logger.PushContext("Reading submesh - ", subMeshName, ":");
            auto mh = ::Mesh::empty(subMeshName);
            mh.topology = vk::PrimitiveTopology::eTriangleList;

            auto contains = [&](std::string name) {
                return std::find_if(primitive.attributes.begin(), primitive.attributes.end(), [&](const auto& x) { return x.first == name; }) != primitive.attributes.end();
            };
            auto getIndex = [&](std::string name) {
                return (uint32_t)std::find_if(primitive.attributes.begin(), primitive.attributes.end(), [&](const auto& x) { return x.first == name; })->second;
            };

            if (!contains("POSITION")) {
                logger.LogError("Mesh data is missing vertices.");
                continue;
            }

            bool hasTexcoords = contains("TEXCOORD_0");
            bool hasNormals = contains("NORMAL");
            bool hasTangents = contains("TANGENT");
            bool hasIndices = primitive.indices >= 0;

            auto& vertexAcc = model.accessors[getIndex("POSITION")];
            auto& vertexBv = model.bufferViews[(size_t)vertexAcc.bufferView];

            auto texcoordAcc = hasTexcoords ? &model.accessors[getIndex("TEXCOORD_0")] : nullptr;
            auto texcoordBv = hasTexcoords ? &model.bufferViews[(size_t)texcoordAcc->bufferView] : nullptr;

            auto work = [&](auto indexIndexer, auto indexCount) {
                auto fromVertexIndexer = makeIndexedIndexer(Indexer<glm::vec3>::make(model.buffers[(size_t)vertexBv.buffer].data.data() + vertexAcc.byteOffset + vertexBv.byteOffset, (uint32_t)vertexBv.byteStride), indexIndexer);

                auto txcoordTypeConverter = [&](void* ptr) -> glm::vec2 {
                    switch (texcoordAcc->componentType) {
                        case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                            return *(glm::vec2*)ptr;
                            break;
                        } case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                            auto& d = *(glm::u8vec2*)ptr;
                            return (glm::vec2)d / 256.f;
                        } case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                            auto& d = *(glm::u16vec2*)ptr;
                            return (glm::vec2)d / 65535.f;
                        } default:
                            assert(false && "Unsupported weight component type");
                            return {};
                    }
                };

                std::optional<IndexedIndexer<Indexer<glm::vec3>, decltype(indexIndexer)>> fromNormalIndexer = std::nullopt;
                if (hasNormals) {
                    auto& normalAcc = model.accessors[getIndex("NORMAL")];
                    auto& normalBv = model.bufferViews[(size_t)normalAcc.bufferView];
                    fromNormalIndexer = makeIndexedIndexer(Indexer<glm::vec3>::make(model.buffers[(size_t)normalBv.buffer].data.data() + normalAcc.byteOffset + normalBv.byteOffset, (uint32_t)normalBv.byteStride), indexIndexer);
                }

                std::optional<IndexedIndexer<Indexer<glm::vec4>, decltype(indexIndexer)>> fromTangentIndexer = std::nullopt;
                if (hasTangents) {
                    auto& tangentAcc = model.accessors[getIndex("TANGENT")];
                    auto& tangentBv = model.bufferViews[(size_t)tangentAcc.bufferView];
                    fromTangentIndexer = makeIndexedIndexer(Indexer<glm::vec4>::make(model.buffers[(size_t)tangentBv.buffer].data.data() + tangentAcc.byteOffset + tangentBv.byteOffset, (uint32_t)tangentBv.byteStride), indexIndexer);
                }

                if (skin) {
                    auto& weightsAcc = model.accessors[getIndex("WEIGHTS_0")];
                    auto& weightsBv = model.bufferViews[(size_t)weightsAcc.bufferView];

                    auto& jointsAcc = model.accessors[getIndex("JOINTS_0")];
                    auto& jointsBv = model.bufferViews[(size_t)jointsAcc.bufferView];

                    auto weightTypeConverter = [&](void* ptr) -> glm::vec4 {
                        switch (weightsAcc.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                return *(glm::vec4*)ptr;
                                break;
                            } case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                                auto& d = *(glm::u8vec4*)ptr;
                                return (glm::vec4)d / 256.f;
                            } case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                                auto& d = *(glm::u16vec4*)ptr;
                                return (glm::vec4)d / 65535.f;
                            } default:
                                assert(false && "Unsupported weight component type");
                                return {};
                        }
                    };
                    auto fromWeightsIndexer = makeIndexedIndexer(makeConvertingIndexer(model.buffers[(size_t)weightsBv.buffer].data.data() + weightsAcc.byteOffset + weightsBv.byteOffset, (uint32_t)weightsBv.byteStride, weightTypeConverter), indexIndexer);

                    auto jointTypeConverter = [&](void* ptr) {
                        switch (jointsAcc.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                                auto& d = *(glm::u8vec4*)ptr;
                                return (uint32_t)d.x + ((uint32_t)d.y << 8) + ((uint32_t)d.z << 16) + ((uint32_t)d.w << 24);
                            } case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                                auto d = (glm::u8vec4)*(glm::u16vec4*)ptr;
                                return (uint32_t)d.x + ((uint32_t)d.y << 8) + ((uint32_t)d.z << 16) + ((uint32_t)d.w << 24);
                            } default:
                                assert(false && "Unsupported joint component type");
                                return 0u;
                        }
                    };
                    auto fromJointsIndexer = makeIndexedIndexer(makeConvertingIndexer(model.buffers[(size_t)jointsBv.buffer].data.data() + jointsAcc.byteOffset + jointsBv.byteOffset, (uint32_t)jointsBv.byteStride, jointTypeConverter), indexIndexer);

                    if (hasTexcoords) {
                        int stride = texcoordAcc->ByteStride(*texcoordBv);
                        assert(stride > 0);
                        auto fromTexcoordIndexer = makeIndexedIndexer(makeConvertingIndexer(model.buffers[(size_t)texcoordBv->buffer].data.data() + texcoordAcc->byteOffset + texcoordBv->byteOffset, (uint32_t)stride, txcoordTypeConverter), indexIndexer);
                        generateSkinnedMesh(fromVertexIndexer, fromTexcoordIndexer, fromNormalIndexer, fromTangentIndexer, fromJointsIndexer, fromWeightsIndexer, indexCount, mh);
                    } else {
                        generateSimpleSkinnedMesh(fromVertexIndexer, fromNormalIndexer, fromJointsIndexer, fromWeightsIndexer, indexCount, mh);
                    }
                } else {
                    if (hasTexcoords) {
                        int stride = texcoordAcc->ByteStride(*texcoordBv);
                        assert(stride > 0);
                        auto fromTexcoordIndexer = makeIndexedIndexer(makeConvertingIndexer(model.buffers[(size_t)texcoordBv->buffer].data.data() + texcoordAcc->byteOffset + texcoordBv->byteOffset, (uint32_t)stride, txcoordTypeConverter), indexIndexer);
                        generateMesh(fromVertexIndexer, fromTexcoordIndexer, fromNormalIndexer, fromTangentIndexer, indexCount, mh);
                    } else {
                        generateSimpleMesh(fromVertexIndexer, fromNormalIndexer, indexCount, mh);
                    }
                }
            };

            if (hasIndices) {
                auto& indexAcc = model.accessors[(size_t)primitive.indices];
                auto& indexBv = model.bufferViews[(size_t)indexAcc.bufferView];
                auto indexComp = GetComponentSizeInBytes((uint32_t)indexAcc.componentType);

                if (indexComp == 2) {
                    auto indexIndexer = Indexer<short>::make(model.buffers[(size_t)indexBv.buffer].data.data() + indexAcc.byteOffset + indexBv.byteOffset, 0);
                    work(indexIndexer, (uint32_t)indexAcc.count);
                } else if (indexComp == 4) {
                    auto indexIndexer = Indexer<int>::make(model.buffers[(size_t)indexBv.buffer].data.data() + indexAcc.byteOffset + indexBv.byteOffset, 0);
                    work(indexIndexer, (uint32_t)indexAcc.count);
                } else {
                    assert(false && "Unsupported index type while reading GLTF model.");
                }
            } else {
                work(DummyIndexer{}, (uint32_t)vertexAcc.count);
            }

            auto colmh = ColouredMesh::empty(meshName);
            colmh.mesh = am.add(std::move(mh));
            colmh.material = materials[(size_t)primitive.material];
            colmh.rendererDependencyIndex = theRendererDependencyProvider.addDependency(RendererDependency{ colmh.mesh->layout, colmh.material->hasTransparency });

            cmh.meshHandlers.push_back({getMatrix(N), am.add(std::move(colmh))});
        }
    }
}

void AssetLoader::validateGlbFile(CompositeMesh& cmh, CompatibilityDescriptor compat)
{
    TinyGLTF tg;
    Model model;
    std::string err, warn;
    tg.LoadBinaryFromFile(&model, &err, &warn, cmh.path);
    loadGltfData(cmh, compat, model);
}

void AssetLoader::validateGltfFile(CompositeMesh& cmh, CompatibilityDescriptor compat)
{
    TinyGLTF tg;
    Model model;
    std::string err, warn;
    tg.LoadASCIIFromFile(&model, &err, &warn, cmh.path);
    loadGltfData(cmh, compat, model);
}

void ImageResource::calculatePixelFormat(int pixelType)
{
    /*
      if (componentType == TINYGLTF_COMPONENT_TYPE_BYTE) {
        return 1;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        return 1;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_SHORT) {
        return 2;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        return 2;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_INT) {
        return 4;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        return 4;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return 4;
      } else if (componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE) {
        return 8;
      } else {
        // Unknown componenty type
        return -1;
      }
    */

    std::pair<vk::Format, vk::Format> formatPair;
    if (channel == 1 && pixelType == TINYGLTF_COMPONENT_TYPE_BYTE)
        formatPair = { vk::Format::eR8Snorm, vk::Format::eR8Srgb };
    else if (channel == 2 && pixelType == TINYGLTF_COMPONENT_TYPE_BYTE)
        formatPair = { vk::Format::eR8G8Snorm, vk::Format::eR8G8Srgb };
    else if (channel == 4 && pixelType == TINYGLTF_COMPONENT_TYPE_BYTE)
        formatPair = { vk::Format::eR8G8B8A8Snorm, vk::Format::eR8G8B8A8Srgb };

    else if (channel == 1 && pixelType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        formatPair = { vk::Format::eR8Unorm, vk::Format::eR8Srgb };
    else if (channel == 2 && pixelType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        formatPair = { vk::Format::eR8G8Unorm, vk::Format::eR8G8Srgb };
    else if (channel == 4 && pixelType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        formatPair = { vk::Format::eR8G8B8A8Unorm, vk::Format::eR8G8B8A8Srgb };

    else assert(false && "Unhandled pixel format!");

    pixelFormat = sRGB ? formatPair.second : formatPair.first;
}

void ImageResource::calculateTransparency()
{
    if (pixelFormat == vk::Format::eR8G8B8A8Unorm || pixelFormat == vk::Format::eR8G8B8A8Srgb) {
        for (uint32_t i = 0; i < width * height; ++i) {
            auto pixelAlpha = data[4 * i + 3];
            if ((unsigned char)pixelAlpha < 255) {
                transparent = true;
                break;
            }
        }
    }
}
