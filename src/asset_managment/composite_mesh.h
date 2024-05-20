#ifndef VULKAN_INTRO_COMPOSITE_MESH_H
#define VULKAN_INTRO_COMPOSITE_MESH_H

#include "coloured_mesh.h"

struct CompositeMeshData {
    glm::mat4 transform;
    ColouredMeshHandler mesh;

    bool operator==(const CompositeMeshData& that) const { return mesh == that.mesh; }
    bool operator!=(const CompositeMeshData& that) const { return !(*this == that); }
};

std::tuple<
    std::array<vk::Buffer, (int)VertexDataElem::_SIZE>,
    std::array<vk::DeviceSize, (int)VertexDataElem::_SIZE>,
    uint32_t>
    extractVertexBufferBindingInfo(
        const std::array<VertexBufferBindingInfo, (int)VertexDataElem::_SIZE>& infos,
        const std::array<VertexDataElem,          (int)VertexDataElem::_SIZE>& bindings
    );

struct CompositeMesh : public AssetBase {
    static CompositeMesh empty(std::string name);
    static std::string typeName();

    void prepare();
    void destroy();
    bool ready() const;
    bool valid() const;

    bool operator==(const CompositeMesh& that) const;
    bool operator!=(const CompositeMesh& that) const;

    glm::mat4 getTransform(size_t index) const;
    AABB getAABB(size_t index) const;
    std::array<VertexBufferBindingInfo, (int)VertexDataElem::_SIZE> getVertexBindingInfo(size_t index) const;
    IndexBufferBindingInfo getIndexBindingInfo(size_t index) const;
    uint32_t getIndexCount(size_t index) const;
    size_t getDependencyIndex(size_t index) const;
    int getMaterialIndex(size_t index) const;

public:
    std::vector<CompositeMeshData> meshHandlers;
    std::string path;

private:
    CompositeMesh() = default;
    void generateBlas();
};

using CompositeMeshHandler = AssetHandler<CompositeMesh>;

template<> CompositeMesh* AssetHandler<CompositeMesh>::operator->();

#endif//VULKAN_INTRO_COMPOSITE_MESH_H
