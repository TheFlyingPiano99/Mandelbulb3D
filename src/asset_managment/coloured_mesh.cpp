#include "coloured_mesh.h"
#include "vulkan_layer.h"
#include "rendering/renderer_dependency_provider.h"

void ColouredMesh::prepare()
{
    mesh->prepare();
    material->prepare();
}

bool ColouredMesh::ready() const
{
    bool rdy = true;
    rdy &= mesh->ready() && material->ready();
    return rdy;
}

bool ColouredMesh::valid() const
{
    bool valid = true;
    valid &= mesh.valid() && material.valid();
    return valid;
}

bool ColouredMesh::operator==(const ColouredMesh& that) const {
    return material == that.material && mesh == that.mesh;
}

bool ColouredMesh::operator!=(const ColouredMesh& that) const {
    return !(*this == that);
}

ColouredMesh ColouredMesh::empty(std::string name)
{
    ColouredMesh cmh;
    cmh.mName = std::move(name);
    return cmh;
}

std::string ColouredMesh::typeName() {
    return "Coloured Mesh";
}
