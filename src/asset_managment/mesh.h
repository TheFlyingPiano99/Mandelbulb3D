#ifndef VULKAN_INTRO_MESH_H
#define VULKAN_INTRO_MESH_H

#include "asset_base.hpp"
#include "mesh_traits.h"
#include "resources/resource_manager.hpp"
#include "utility/mmath.h"

struct VertexBufferBindingInfo {
    BufferResourceHandler buffer;
    vk::DeviceSize offset;

    bool operator==(const VertexBufferBindingInfo& that) const { return buffer == that.buffer && offset == that.offset; }
    bool operator!=(const VertexBufferBindingInfo& that) const { return buffer != that.buffer || offset != that.offset; }
};

struct IndexBufferBindingInfo {
    BufferResourceHandler buffer;
    vk::DeviceSize offset;
    vk::IndexType type;

    bool operator==(const IndexBufferBindingInfo& that) const { return buffer == that.buffer && offset == that.offset && type == that.type; }
    bool operator!=(const IndexBufferBindingInfo& that) const { return buffer != that.buffer || offset != that.offset || type != that.type; }

    void setTypeBySize(int size) {
        switch (size) {
            case 2: type = vk::IndexType::eUint16; break;
            case 4: type = vk::IndexType::eUint32; break;
            default: logger.LogError("Invalid index buffer size!"); type = vk::IndexType::eUint16; break;         
        }
    }
};

struct Mesh : public AssetBase {
    static Mesh empty(std::string name);
    static std::string typeName();

    void prepare();
    bool valid() const;
    bool ready() const;

    bool operator==(const Mesh& that) const;
    bool operator!=(const Mesh& that) const;

    uint32_t indexCount;
    vk::PrimitiveTopology topology;
    glm::vec3 parentTranslation;
    bool animation;

    MeshDataLayout layout;

    IndexBufferBindingInfo indexBuffer;
    std::array<VertexBufferBindingInfo, (int)VertexDataElem::_SIZE> vertexDataBuffers;

    AABB aabb;

private:
    Mesh() = default;
};

using MeshHandler = AssetHandler<Mesh>;

template<> Mesh* AssetHandler<Mesh>::operator->();

#endif//VULKAN_INTRO_MESH_H
