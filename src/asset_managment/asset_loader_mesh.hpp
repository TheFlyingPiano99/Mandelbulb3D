#ifndef VULKAN_INTRO_ASSET_LOADER_MESH_HPP
#define VULKAN_INTRO_ASSET_LOADER_MESH_HPP

#include "utility/utility.hpp"
#include "mesh.h"
#include "constants.h"

template <typename VertexIndexer>
void generateFlatNormals(VertexIndexer positions, size_t indexCount, Indexer<glm::vec3> normals)
{
    for (size_t i = 0; i <= indexCount - 3; i+= 3) {
        auto v1 = positions[i], v2 = positions[i + 1], v3 = positions[i + 2];
        normals[i] = normals[i+1] = normals[i+2] = glm::normalize(glm::cross(v2 - v1, v3 - v1));
    }
}

template <typename VertexIndexer, typename TexcoordIndexer, typename NormalIndexer>
void generateSimpleTangents(VertexIndexer positions, TexcoordIndexer texcoords, NormalIndexer, size_t indexCount, Indexer<glm::vec4> tangents)
{
    for (size_t i = 0; i <= indexCount - 3; i+= 3) {
        auto v1 = positions[i], v2 = positions[i + 1], v3 = positions[i + 2];
        auto vt1 = texcoords[i], vt2 = texcoords[i + 1], vt3 = texcoords[i + 2];

        glm::vec3 edge1 = v2 - v1;
        glm::vec3 edge2 = v3 - v1;
        glm::vec2 deltaUV1 = vt2 - vt1;
        glm::vec2 deltaUV2 = vt3 - vt1;

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
        
        glm::vec3 tangent;
        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        tangent = glm::normalize(tangent);
        tangents[i] = tangents[i + 1] = tangents[i + 2] = glm::vec4{tangent.x, tangent.y, tangent.z, 1.0};// TODO: left or right! handed?
    }
}

template <typename PositionIndexer>
AABB calculateAABB(PositionIndexer positions, size_t count)
{
    AABB aabb;
    aabb.tr = aabb.bl = positions[0];
    for (size_t i = 0; i < count; ++i) {
        aabb.bl = glm::min(aabb.bl, positions[i]);
        aabb.tr = glm::max(aabb.tr, positions[i]);
    }
    return aabb;
}

template <typename VertexIndexer, typename NormalIndexer>
void generateSimpleMesh(VertexIndexer& fromPositions, std::optional<NormalIndexer>& fromNormals, uint32_t indexCount, ::Mesh& mesh)
{
    struct VertexData {
        glm::vec3 position, normal;
    };

    std::vector<std::byte> data;
    data.resize(indexCount * sizeof(VertexData));

    auto vertexIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, position)), sizeof(VertexData));
    copy(fromPositions, indexCount, vertexIndexer);

    auto normalIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, normal)), sizeof(VertexData));
    if (!fromNormals) {
        generateFlatNormals(vertexIndexer, indexCount, normalIndexer);
    } else {
        copy(fromNormals.value(), indexCount, normalIndexer);
    }

    BufferResource buf;
    buf.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;

    buf.data = std::move(data);
    buf.mLocation = ResourceLocation::InMemoryBit;

    buf.upload();

    auto bh = theResourceManager.addBuffer(std::move(buf));

    mesh.indexCount = indexCount;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].buffer = bh;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].offset = offsetof(VertexData, position);
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].offset = offsetof(VertexData, normal);

    int vertexDataSiz = sizeof(VertexData);

    mesh.layout.infos[VertexDataElem::Position] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Normal] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.indexInfo = {0, 0, 0};

    mesh.aabb = calculateAABB(fromPositions, indexCount);

    if (cLogAssetLoading) logger.LogInfo("Loaded mesh with ", mesh.indexCount, " vertices.");
}

template <typename VertexIndexer, typename NormalIndexer, typename WeightIndexer, typename JointIndexer>
void generateSimpleSkinnedMesh(VertexIndexer& fromPositions, std::optional<NormalIndexer>& fromNormals, JointIndexer& fromJoints, WeightIndexer& fromWeights, uint32_t indexCount, Mesh& mesh)
{
    struct VertexData {
        glm::vec3 position, normal;
        glm::vec4 weights;
        uint32_t joints;
    };

    std::vector<std::byte> data;
    data.resize(indexCount * sizeof(VertexData));

    auto vertexIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, position)), sizeof(VertexData));
    copy(fromPositions, indexCount, vertexIndexer);

    auto weightIndexer = Indexer<glm::vec4>::make((void*)(data.data() + offsetof(VertexData, weights)), sizeof(VertexData));
    copy(fromWeights, indexCount, weightIndexer);

    auto jointIndexer = Indexer<uint32_t>::make((void*)(data.data() + offsetof(VertexData, joints)), sizeof(VertexData));
    copy(fromJoints, indexCount, jointIndexer);

    auto normalIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, normal)), sizeof(VertexData));
    if (!fromNormals) {
        generateFlatNormals(vertexIndexer, indexCount, normalIndexer);
    } else {
        copy(fromNormals.value(), indexCount, normalIndexer);
    }

    BufferResource buf;
    buf.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;

    buf.data = std::move(data);
    buf.mLocation = ResourceLocation::InMemoryBit;

    buf.upload();

    auto bh = theResourceManager.addBuffer(std::move(buf));

    mesh.indexCount = indexCount;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Weights].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Joints].buffer = bh;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].offset = offsetof(VertexData, position);
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].offset = offsetof(VertexData, normal);
    mesh.vertexDataBuffers[(int)VertexDataElem::Weights].offset = offsetof(VertexData, weights);
    mesh.vertexDataBuffers[(int)VertexDataElem::Joints].offset = offsetof(VertexData, joints);

    int vertexDataSiz = sizeof(VertexData);

    mesh.layout.infos[VertexDataElem::Position] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Normal] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Weights] = {(unsigned short)vertexDataSiz, 4, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Joints] = {(unsigned short)vertexDataSiz, 1, sizeof(uint32_t)};
    mesh.layout.indexInfo = { 0, 0, 0 };

    mesh.aabb = calculateAABB(fromPositions, indexCount);

    if (cLogAssetLoading) logger.LogInfo("Loaded mesh with ", mesh.indexCount, " vertices.");
}

template <typename VertexIndexer, typename TexcoordIndexer, typename NormalIndexer, typename TangentIndexer>
void generateMesh(VertexIndexer& fromPositions, TexcoordIndexer& fromTexcoords, std::optional<NormalIndexer>& fromNormals, std::optional<TangentIndexer>& fromTangents, uint32_t indexCount, ::Mesh& mesh)
{
    struct VertexData {
        glm::vec3 position, normal;
        glm::vec4 tangent;
        glm::vec2 texcoord;
    };

    std::vector<std::byte> data;
    data.resize(indexCount * sizeof(VertexData));

    auto vertexIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, position)), sizeof(VertexData));
    copy(fromPositions, indexCount, vertexIndexer);

    auto texcoordIndexer = Indexer<glm::vec2>::make((void*)(data.data() + offsetof(VertexData, texcoord)), sizeof(VertexData));
    copy(fromTexcoords, indexCount, texcoordIndexer);

    auto normalIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, normal)), sizeof(VertexData));
    if (!fromNormals) {
        generateFlatNormals(vertexIndexer, indexCount, normalIndexer);
    } else {
        copy(fromNormals.value(), indexCount, normalIndexer);
    }

    auto tangentIndexer = Indexer<glm::vec4>::make((void*)(data.data() + offsetof(VertexData, tangent)), sizeof(VertexData));
    if (!fromTangents) {
        generateSimpleTangents(vertexIndexer, texcoordIndexer, normalIndexer, indexCount, tangentIndexer);
    } else {
        copy(fromTangents.value(), indexCount, tangentIndexer);
    }

    BufferResource buf;
    buf.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;

    buf.data = std::move(data);
    buf.mLocation = ResourceLocation::InMemoryBit;

    buf.upload();

    auto bh = theResourceManager.addBuffer(std::move(buf));

    mesh.indexCount = indexCount;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Tangent].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::TextureCoordinate].buffer = bh;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].offset = offsetof(VertexData, position);
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].offset = offsetof(VertexData, normal);
    mesh.vertexDataBuffers[(int)VertexDataElem::Tangent].offset = offsetof(VertexData, tangent);
    mesh.vertexDataBuffers[(int)VertexDataElem::TextureCoordinate].offset = offsetof(VertexData, texcoord);

    int vertexDataSiz = sizeof(VertexData);

    mesh.layout.infos[VertexDataElem::Position] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Normal] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Tangent] = {(unsigned short)vertexDataSiz, 4, sizeof(float)};
    mesh.layout.infos[VertexDataElem::TextureCoordinate] = {(unsigned short)vertexDataSiz, 2, sizeof(float)};
    mesh.layout.indexInfo = { 0, 0, 0 };

    mesh.aabb = calculateAABB(fromPositions, indexCount);

    if (cLogAssetLoading) logger.LogInfo("Loaded mesh with ", mesh.indexCount, " vertices.");
}

template <typename VertexIndexer, typename TexcoordIndexer, typename NormalIndexer, typename TangentIndexer, typename WeightIndexer, typename JointIndexer>
void generateSkinnedMesh(VertexIndexer& fromPositions, TexcoordIndexer& fromTexcoords, std::optional<NormalIndexer>& fromNormals, std::optional<TangentIndexer>& fromTangents, JointIndexer& fromJoints, WeightIndexer& fromWeights, uint32_t indexCount, ::Mesh& mesh)
{
    struct VertexData {
        glm::vec3 position, normal;
        glm::vec4 tangent;
        glm::vec2 texcoord;
        glm::vec4 weights;
        uint32_t joints;
    };

    std::vector<std::byte> data;
    data.resize(indexCount * sizeof(VertexData));

    auto vertexIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, position)), sizeof(VertexData));
    copy(fromPositions, indexCount, vertexIndexer);

    auto texcoordIndexer = Indexer<glm::vec2>::make((void*)(data.data() + offsetof(VertexData, texcoord)), sizeof(VertexData));
    copy(fromTexcoords, indexCount, texcoordIndexer);

    auto weightIndexer = Indexer<glm::vec4>::make((void*)(data.data() + offsetof(VertexData, weights)), sizeof(VertexData));
    copy(fromWeights, indexCount, weightIndexer);

    auto jointIndexer = Indexer<uint32_t>::make((void*)(data.data() + offsetof(VertexData, joints)), sizeof(VertexData));
    copy(fromJoints, indexCount, jointIndexer);

    auto normalIndexer = Indexer<glm::vec3>::make((void*)(data.data() + offsetof(VertexData, normal)), sizeof(VertexData));
    if (!fromNormals) {
        generateFlatNormals(vertexIndexer, indexCount, normalIndexer);
    } else {
        copy(fromNormals.value(), indexCount, normalIndexer);
    }

    auto tangentIndexer = Indexer<glm::vec4>::make((void*)(data.data() + offsetof(VertexData, tangent)), sizeof(VertexData));
    if (!fromTangents) {
        generateSimpleTangents(vertexIndexer, texcoordIndexer, normalIndexer, indexCount, tangentIndexer);
    } else {
        copy(fromTangents.value(), indexCount, tangentIndexer);
    }

    BufferResource buf;
    buf.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;

    buf.data = std::move(data);
    buf.mLocation = ResourceLocation::InMemoryBit;

    buf.upload();

    auto bh = theResourceManager.addBuffer(std::move(buf));

    mesh.indexCount = indexCount;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Tangent].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::TextureCoordinate].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Weights].buffer = bh;
    mesh.vertexDataBuffers[(int)VertexDataElem::Joints].buffer = bh;

    mesh.vertexDataBuffers[(int)VertexDataElem::Position].offset = offsetof(VertexData, position);
    mesh.vertexDataBuffers[(int)VertexDataElem::Normal].offset = offsetof(VertexData, normal);
    mesh.vertexDataBuffers[(int)VertexDataElem::Tangent].offset = offsetof(VertexData, tangent);
    mesh.vertexDataBuffers[(int)VertexDataElem::TextureCoordinate].offset = offsetof(VertexData, texcoord);
    mesh.vertexDataBuffers[(int)VertexDataElem::Weights].offset = offsetof(VertexData, weights);
    mesh.vertexDataBuffers[(int)VertexDataElem::Joints].offset = offsetof(VertexData, joints);

    int vertexDataSiz = sizeof(VertexData);

    mesh.layout.infos[VertexDataElem::Position] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Normal] = {(unsigned short)vertexDataSiz, 3, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Tangent] = {(unsigned short)vertexDataSiz, 4, sizeof(float)};
    mesh.layout.infos[VertexDataElem::TextureCoordinate] = {(unsigned short)vertexDataSiz, 2, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Weights] = {(unsigned short)vertexDataSiz, 4, sizeof(float)};
    mesh.layout.infos[VertexDataElem::Joints] = {(unsigned short)vertexDataSiz, 1, sizeof(uint32_t)};
    mesh.layout.indexInfo = { 0, 0, 0 };

    mesh.aabb = calculateAABB(fromPositions, indexCount);

    if (cLogAssetLoading) logger.LogInfo("Loaded mesh with ", mesh.indexCount, " vertices.");
}

#endif//VULKAN_INTRO_ASSET_LOADER_MESH_HPP
