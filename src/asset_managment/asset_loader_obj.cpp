#include "asset_loader.h"
#include "asset_loader_mesh.hpp"
#include "asset_loader_texture.h"
#include "asset_manager.h"
#include "rendering/renderer_dependency_provider.h"
#include "utility/utility.hpp"
#include <tinyobjloader/tiny_obj_loader.h>

namespace fs = std::filesystem;

bool collectPosition(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index, glm::vec3* v)
{
    if (index.vertex_index >= 0) {
        std::memcpy(v, &attrib.vertices[3 * (size_t)index.vertex_index], 3 * 4);
        return true;
    }
    return false;
}

bool collectNormal(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index, glm::vec3* vn)
{
    if (index.normal_index >= 0) {
        std::memcpy(vn, &attrib.normals[3 * (size_t)index.normal_index], 3 * 4);
        return true;
    }
    return false;
}

bool collectTextureCoordinate(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index, glm::vec2* vt)
{
    if (index.texcoord_index >= 0) {
        std::memcpy(vt, &attrib.texcoords[2 * (size_t)index.texcoord_index], 2 * 4);
        return true;
    }
    return false;
}

bool readFromTinyObjLoaderThings(Mesh& mesh, const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape)
{
    auto& inds = shape.mesh.indices;

    if (attrib.vertices.empty()) {
        logger.LogError("Mesh data is missing vertices.");
        return false;
    }

    if (attrib.texcoords.empty()) {
        logger.LogError("Mesh data is missing texture coordinates.");
        return false;
    }

    auto fromVertexIndexer = makeIndexedIndexer(Indexer<glm::vec3>::make((void*)attrib.vertices.data(), 0), Indexer<int>::make((void*)&inds[0].vertex_index, sizeof(inds[0])));
    auto fromTexcoordIndexer = makeIndexedIndexer(Indexer<glm::vec2>::make((void*)attrib.texcoords.data(), 0), Indexer<int>::make((void*)&inds[0].texcoord_index, sizeof(inds[0])));
    std::optional<IndexedIndexer<Indexer<glm::vec3>, Indexer<int>>> fromNormalIndexer = std::nullopt;

    if (!attrib.normals.empty()) {
        fromNormalIndexer = makeIndexedIndexer(Indexer<glm::vec3>::make((void*)attrib.normals.data(), 0), Indexer<int>::make((void*)&inds[0].normal_index, sizeof(inds[0])));
    }

    std::optional<IndexedIndexer<Indexer<glm::vec4>, Indexer<int>>> fromTangentIndexer = std::nullopt;
    generateMesh(fromVertexIndexer, fromTexcoordIndexer, fromNormalIndexer, fromTangentIndexer, (uint32_t)inds.size(), mesh);

    return true;
}

void AssetLoader::validateObjFile(CompositeMesh& cmh, CompatibilityDescriptor compat)
{
    auto& path = cmh.path;
    const auto& name = cmh.name();

    auto mtlDir = path.substr(0, path.rfind('/'));

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    auto _l2 = logger.PushContext("Loading ", path, ":");
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), mtlDir.c_str())) {
        logger.LogError("\tError: ", err, ".\n\tWarn: ", warn, ".");
        return;
    }

    std::vector<MaterialHandler> mhs;

    bool hasDiffuse = false, hasMro = false;

    for (auto& material : materials) {
        auto _l3 = logger.PushContext("Loading material: ", material.name);
        auto mh = Material::empty(std::string{name} + "/" + material.name);
        for (auto& desc : compat.materialDescriptor.textureDescriptors) {
            std::vector<std::pair<std::string, std::string>> textures;
            switch (desc.type) {
                case MaterialElemTypeTexture::DiffuseTexture:
                    textures.emplace_back(material.diffuse_texname, material.diffuse_texopt.colorspace);
                    hasDiffuse = true;
                    break;
                case MaterialElemTypeTexture::NormalTexture:
                    textures.emplace_back(material.normal_texname, material.normal_texopt.colorspace);
                    break;
                case MaterialElemTypeTexture::CombinedMRO:
                    textures.emplace_back(material.metallic_texname, material.metallic_texopt.colorspace);
                    textures.emplace_back(material.roughness_texname, material.roughness_texopt.colorspace);
                    textures.emplace_back(material.ambient_texname, material.ambient_texopt.colorspace);
                    hasMro = true;
                    break;
                default:
                    assert(false && "Unhandled material texture type!");
            }

            TextureHandler txh;
            bool invalid = false, dummy = false;
            for (auto& it : textures) {
                if (it.first.empty()) {
                    dummy = true;
                } else {
                    bool inv2 = false;
                    for (auto& root : rootPaths) {
                        inv2 = true;
                        if (fs::is_regular_file(root + "/" + it.first)) {
                            it.first = root + "/" + it.first;
                            inv2 = false;
                            break;
                        }
                    }
                    if (inv2) {
                        if (fs::is_regular_file(mtlDir + "/" + it.first)) {
                            it.first = mtlDir + "/" + it.first;
                            inv2 = false;
                        } else {
                            logger.LogError("Required texture was not found on disk: ",  it.first);
                            invalid = true;
                        }
                    }
                }
            }
            if (invalid) {
                txh = am.add(Texture::empty(std::string{name} + "/" + material.name + "/" + std::to_string((int) desc.type)));
            } else {
                if (dummy) {
                    txh = am.add(Texture::dummy(std::string{name} + "/" + material.name + "/" + std::to_string((int) desc.type)));
                } else if (textures.size() == 1) {
                    txh = handleTexture(am, textures[0].first, textures[0].second, desc.sRGB, desc.mipMapped);
                } else {
                    txh = handleTextureCombination(am, textures, desc.mipMapped);
                }

                txh->prepare();
            }

            mh.textures.push_back(txh);
        }

        for (auto& desc : compat.materialDescriptor.valueDescriptors) {
            float value;
            switch (desc) {
                case MaterialElemTypeValue::DiffuseValueR:
                    value = !hasDiffuse ? 1.0 : material.diffuse[0];
                    break;
                case MaterialElemTypeValue::DiffuseValueG:
                    value = !hasDiffuse ? 1.0 : material.diffuse[1];
                    break;
                case MaterialElemTypeValue::DiffuseValueB:
                    value = !hasDiffuse ? 1.0 : material.diffuse[2];
                    break;
                case MaterialElemTypeValue::OpacityValueR:
                    value = !hasDiffuse ? 1.0 : material.dissolve;
                    break;
                case MaterialElemTypeValue::RoughnessValueR:
                    value = !hasMro ? 1.0 : material.roughness;
                    break;
                case MaterialElemTypeValue::MetallicValueR:
                    value = !hasMro ? 1.0 : material.metallic;
                    break;
                default:
                    assert(false && "Unhandled material value type!");
            }

            mh.values.push_back(value);
        }

        mhs.emplace_back(am.add(std::move(mh)));
    }

    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto& subMeshName = shapes[i].name.empty() ? std::to_string(i) : shapes[i].name;
        auto _l4 = logger.PushContext("Reading submesh - ", subMeshName, ":");
        auto mh = Mesh::empty(path + "/" + subMeshName);

        mh.topology = vk::PrimitiveTopology::eTriangleList;
        mh.parentTranslation = {0, 0, 0};

        if (!readFromTinyObjLoaderThings(mh, attrib, shapes[i])) {
            continue;
        }

        auto pMesh = am.add(std::move(mh));
        pMesh->prepare();

        MaterialHandler pmh;
        // per-face materials are not supported
        if (!shapes[i].mesh.material_ids.empty() && shapes[i].mesh.material_ids[0] > -1) {
            pmh = mhs[(size_t)shapes[i].mesh.material_ids[0]];
        } else {
            auto missingMat = Material::empty(path + "/missingMaterial");
            pmh = am.add(std::move(missingMat));
            logger.LogError("Material not found.");
        }
        auto colmh = ColouredMesh::empty(path + "/" + subMeshName);
        colmh.mesh = pMesh;
        colmh.material = pmh;
        colmh.rendererDependencyIndex = theRendererDependencyProvider.addDependency(RendererDependency{ colmh.mesh->layout, colmh.material->hasTransparency });
            
        auto pTMesh = am.add(std::move(colmh));
        cmh.meshHandlers.push_back({glm::translate(glm::vec3{0, 0, 0}), pTMesh});
    }
}
