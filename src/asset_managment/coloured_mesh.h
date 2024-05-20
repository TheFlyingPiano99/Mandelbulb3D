#ifndef VULKAN_INTRO_COLOURED_MESH_H
#define VULKAN_INTRO_COLOURED_MESH_H

#include "mesh.h"
#include "material.h"

struct ColouredMesh : public AssetBase {
    static ColouredMesh empty(std::string name);
    static std::string typeName();

    void prepare();
    bool ready() const;
    bool valid() const;

    bool operator==(const ColouredMesh& that) const;
    bool operator!=(const ColouredMesh& that) const;

    MeshHandler mesh;
    MaterialHandler material;

    size_t rendererDependencyIndex;

private:
    ColouredMesh() = default;
};

using ColouredMeshHandler = AssetHandler<ColouredMesh>;

template<> ColouredMesh* AssetHandler<ColouredMesh>::operator->();

#endif//VULKAN_INTRO_COLOURED_MESH_H
