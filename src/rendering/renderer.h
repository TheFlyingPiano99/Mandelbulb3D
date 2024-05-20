#include "rendering/render_context.h"
#include "rendering/common_rendering.h"
#include "asset_managment/mesh_traits.h"
#include "constants.h"
#include "shader_manager.h"
#include "utility/utility.hpp"

struct RendererDependency;

class Renderer {
public:
    Renderer() : perFrameDscSetBuilder{ descriptorPoolBuilder } {}
    void initialize();
    void destroy();
    void initializeSwapChainDependentComponents();
    void destroySwapChainDependentComponents();
    void render(const RenderContext& ctx);
    void update(double dt);

private:
    struct DependentComponents {
        vk::Pipeline pipeline = nullptr;
        ShaderDependencyReference shaderRef;
        std::array<VertexDataElem, (int)VertexDataElem::_SIZE> bindings;

        DependentComponents() {
            bindings.fill(VertexDataElem::_SIZE);
        }

        void destroy()
        {
            if (pipeline) {
                theVulkanLayer.device.destroy(pipeline);
                pipeline = nullptr;
                theShaderManager.removeShaderDependency(shaderRef);
            }
        }
    };

    struct PerFrameUniformData {
        glm::mat4 v;
        glm::vec4 ambientLight;
    };

    struct alignas(16) PerObjectUniformData {
        glm::mat4 mv, mvp, normal;
        int materialIndex;
    };

    struct alignas(16) PerPointLightUniformData {
        glm::vec4 position, power;
    };

    static const size_t MAX_POINTLIGHT_COUNT = 50;
    struct PointLightsUBO {
        glm::uvec4 lightCount_pad3;
        PerPointLightUniformData pointLights[MAX_POINTLIGHT_COUNT];
    };

    void createPipelineLayout();
    void createFsPipeline();
    void createSyncObjects();
    void createResources();
    void createImages();
    void updateDescriptorSets();

private:
    ShaderDependencyReference fsPipelineRef;
    vk::Pipeline fsPipeline;
    vk::PipelineLayout fsPipelineLayout;
    VulkanImage colorImage;
    void* colorImageNativeHandle;
    QueryInfo<2> query;
    vk::Sampler fsSampler;
    std::array<VulkanBuffer, MAX_FRAMES_IN_FLIGHT> perFrameBuffers;
    DescriptorPoolBuilder descriptorPoolBuilder;
    DescriptorSetBuilder perFrameDscSetBuilder;
    std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> perFrameDescriptorSets;

    // for vma pool
    // vma allocates memory by default from a standard pool
    // to assign extension structures describing an external allocation we need to create a custom pool
    // the custom pool uses these extension structures for each allocation so they need to be alive as long as the pool is alive
    WindowsSecurityAttributes winSecurityAttributes;
    VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR;
    VkExportMemoryAllocateInfoKHR exportMemoryInfo;
    VmaPool externalPool;

public:
    // cuda - vulkan synchronization
    vk::Semaphore cudaWaitsForVulkanSemaphore, vulkanWaitsForCudaSemaphore;
};