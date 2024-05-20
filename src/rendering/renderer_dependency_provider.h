#ifndef VULKAN_INTRO_RENDERER_DEPENDENCY_PROVIDER
#define VULKAN_INTRO_RENDERER_DEPENDENCY_PROVIDER

#include "asset_managment/mesh_traits.h"

struct RendererDependency {
    MeshDataLayout layout;
    bool transparent;

    bool operator==(const RendererDependency& that) const { return layout == that.layout && transparent == that.transparent; }
};

class RendererDependencyProvider {
public:
    static RendererDependencyProvider& Instance() { static RendererDependencyProvider instance; return instance; }

    size_t addDependency(const RendererDependency& dep) {
        for (size_t i = 0; i < dependencies.size(); ++i) {
            if (dependencies[i] == dep) return i;
        }
        dependencies.push_back(dep);
        return dependencies.size() - 1;
    }

    const RendererDependency& operator[](size_t index) const { 
        assert(index < dependencies.size());
        return dependencies[index];
    }

private:
    RendererDependencyProvider() = default;

private:
    std::vector<RendererDependency> dependencies;
};

inline auto& theRendererDependencyProvider = RendererDependencyProvider::Instance();

#endif//VULKAN_INTRO_RENDERER_DEPENDENCY_PROVIDER
