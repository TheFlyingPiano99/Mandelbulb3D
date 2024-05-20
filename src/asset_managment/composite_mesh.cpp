#include "composite_mesh.h"
#include "rendering/renderer_dependency_provider.h"

std::tuple<
    std::array<vk::Buffer, (int)VertexDataElem::_SIZE>,
    std::array<vk::DeviceSize, (int)VertexDataElem::_SIZE>,
    uint32_t>
    extractVertexBufferBindingInfo(const std::array<VertexBufferBindingInfo, (int)VertexDataElem::_SIZE>& infos, const std::array<VertexDataElem, (int)VertexDataElem::_SIZE>& bindings)
{
    std::tuple<
        std::array<vk::Buffer, (int)VertexDataElem::_SIZE>,
        std::array<vk::DeviceSize, (int)VertexDataElem::_SIZE>,
        uint32_t> ret;
    uint32_t vertexBufferCount = 0;
    for (size_t i = 0; bindings[i] != VertexDataElem::_SIZE && i < bindings.size(); ++i) {
        std::get<0>(ret)[vertexBufferCount] = infos[(int)bindings[i]].buffer->buffer.buffer;
        std::get<1>(ret)[vertexBufferCount] = infos[(int)bindings[i]].offset;
        vertexBufferCount++;
    }
    std::get<2>(ret) = vertexBufferCount;
    return ret;
}

CompositeMesh CompositeMesh::empty(std::string name)
{
    CompositeMesh cmh;
    cmh.mName = std::move(name);
    return cmh;
}

void CompositeMesh::prepare()
{
    for (auto& md : meshHandlers) {
        md.mesh->prepare();
    }
}

void CompositeMesh::destroy()
{
}

bool CompositeMesh::ready() const
{
    for (auto& md : meshHandlers) {
        if (!md.mesh->ready()) return false;
    }
    return true;
}

bool CompositeMesh::valid() const
{
    return !meshHandlers.empty();
}

bool CompositeMesh::operator==(const CompositeMesh& that) const
{
    return meshHandlers == that.meshHandlers;
}

bool CompositeMesh::operator!=(const CompositeMesh& that) const
{
    return !(*this == that);
}

glm::mat4 CompositeMesh::getTransform(size_t index) const
{
    assert(index < meshHandlers.size());
    return meshHandlers[index].transform;
}

AABB CompositeMesh::getAABB(size_t index) const
{
    assert(index < meshHandlers.size());
    if (meshHandlers[index].mesh->mesh.valid()) {
        return meshHandlers[index].mesh->mesh->aabb;
    } else {
        assert(false);
        return AABB{};
    }
}

std::array<VertexBufferBindingInfo, (int)VertexDataElem::_SIZE> CompositeMesh::getVertexBindingInfo(size_t index) const
{
    assert(index < meshHandlers.size());
    return meshHandlers[index].mesh->mesh->vertexDataBuffers;
}

IndexBufferBindingInfo CompositeMesh::getIndexBindingInfo(size_t index) const
{
    assert(index < meshHandlers.size());
    return meshHandlers[index].mesh->mesh->indexBuffer;
}

uint32_t CompositeMesh::getIndexCount(size_t index) const
{
    assert(index < meshHandlers.size());
    return meshHandlers[index].mesh->mesh->indexCount;
}

size_t CompositeMesh::getDependencyIndex(size_t index) const
{
    assert(index < meshHandlers.size());
    return meshHandlers[index].mesh->rendererDependencyIndex;
}

int CompositeMesh::getMaterialIndex(size_t index) const
{
    assert(index < meshHandlers.size());
    return meshHandlers[index].mesh->material->gpuIndex;
}

std::string CompositeMesh::typeName()
{
    return "Composite Mesh";
}
