#ifndef VULKAN_INTRO_MESH_TRAITS_H
#define VULKAN_INTRO_MESH_TRAITS_H

enum class VertexDataElem : unsigned char {
    Position, Normal, Tangent, TextureCoordinate, Joints, Weights
    ,_SIZE
};

struct VertexDataDescriptor {
    std::vector<VertexDataElem> elems;

    bool operator==(const VertexDataDescriptor& that) const { return elems == that.elems; }
    bool operator!=(const VertexDataDescriptor& that) const { return elems != that.elems; }
};

struct MeshDataInfo {
    unsigned short stride;
    unsigned char components, componentSize;

    bool operator<(const MeshDataInfo& that) const
    { 
        return std::tie(stride, components, componentSize) < std::tie(that.stride, that.components, that.componentSize);
    }

    bool operator==(const MeshDataInfo& that) const
    {
        return std::tie(stride, components, componentSize) == std::tie(that.stride, that.components, that.componentSize);
    }
};

struct MeshDataLayout {
    MeshDataInfo indexInfo;
    std::map<VertexDataElem, MeshDataInfo> infos;

    bool operator==(const MeshDataLayout& that) const
    {
        return std::tie(indexInfo, infos) == std::tie(that.indexInfo, that.infos);
    }
};

namespace std {
    template <> struct hash<VertexDataDescriptor>
    {
        size_t operator()(const VertexDataDescriptor& x) const
        {
            std::size_t seed = x.elems.size();
            for(auto& i : x.elems) {
                seed ^= (unsigned long)i + 0x9e3779b9ul + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

#endif//VULKAN_INTRO_MESH_TRAITS_H
