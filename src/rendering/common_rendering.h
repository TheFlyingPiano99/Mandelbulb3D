#ifndef VULKAN_INTRO_COMMON_RENDERING_H
#define VULKAN_INTRO_COMMON_RENDERING_H

#include "vulkan_layer.h"

template <size_t QueryCount>
class QueryInfo {
public:
    void create() {
        vk::QueryPoolCreateInfo createInfo = {};
        createInfo.queryType = vk::QueryType::eTimestamp;
        createInfo.queryCount = QueryCount;
        queryPool = theVulkanLayer.device.createQueryPool(createInfo);
        auto cmd = theVulkanLayer.beginSingleTimeCommands();
        cmd.resetQueryPool(queryPool, 0, QueryCount);
        theVulkanLayer.endSingleTimeCommands(cmd);
    }
    
    void destroy() {
        theVulkanLayer.device.destroy(queryPool);
    }

    void beginFrame(vk::CommandBuffer cmd) {
        if (!queryInFlight) {
            cmd.resetQueryPool(queryPool, 0, QueryCount);
            counter = 0;
        }
    }

    void write(vk::CommandBuffer cmd, vk::PipelineStageFlagBits stage) {
        if (!queryInFlight) {
            cmd.writeTimestamp(stage, queryPool, counter++);
        }
    }

    void endFrame() {
        queryInFlight = true;
    }

    vk::Result query() {
        uint64_t actQueryRes[QueryCount] = {};
        auto qres = theVulkanLayer.device.getQueryPoolResults(queryPool, 0, QueryCount, QueryCount * sizeof(uint64_t), &actQueryRes, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
        if (qres == vk::Result::eSuccess) {
            memcpy(&results, &actQueryRes, QueryCount * sizeof(uint64_t));
            queryInFlight = false;
        }
        return qres;
    }

    uint64_t results[QueryCount];

private:
    vk::QueryPool queryPool;
    bool queryInFlight = false;
    uint32_t counter = 0;
};

class DescriptorSetBuilder;

class DescriptorPoolBuilder {
public:
    void initialize(vk::DescriptorPoolCreateFlags flags = {});
    void create();
    void destroy();
    void reset();

    friend class DescriptorSetBuilder;
    
private:
    uint32_t setCount;
    std::map<vk::DescriptorType, uint32_t> descriptorCounts;
    vk::DescriptorPoolCreateFlags mFlags;
    vk::DescriptorPool mPool;
};

class DescriptorSetBuilder {
public:
    DescriptorSetBuilder(DescriptorPoolBuilder& poolBuilder) : mPoolBuilder{ &poolBuilder } {}
    DescriptorSetBuilder() : mPoolBuilder{ nullptr } {}

    class DescriptorSetUpdater {
    public:
        DescriptorSetUpdater(DescriptorSetBuilder& descSetBuilder, vk::ArrayProxy<vk::DescriptorSet> descriptorSets);
        DescriptorSetUpdater& writeBuffer(uint32_t binding, vk::ArrayProxy<VulkanBuffer> buffers, vk::DeviceSize offset = 0, vk::DeviceSize range = VK_WHOLE_SIZE);
        DescriptorSetUpdater& writeBuffer(uint32_t binding, vk::ArrayProxy<vk::Buffer> buffers, vk::DeviceSize offset = 0, vk::DeviceSize range = VK_WHOLE_SIZE);
        DescriptorSetUpdater& writeImage(uint32_t binding, vk::ArrayProxy<vk::ImageView> images, vk::ImageLayout imageLayout);
        DescriptorSetUpdater& writeSampler(uint32_t binding, vk::ArrayProxy<vk::Sampler> samplers);

    private:
        vk::ArrayProxy<vk::DescriptorSet> descSets;
        std::vector<vk::WriteDescriptorSet> descWrites;
        std::vector<vk::DescriptorImageInfo> imageInfos;
        std::vector<vk::DescriptorBufferInfo> bufferInfos;
        std::vector<vk::WriteDescriptorSetAccelerationStructureKHR> tlasInfos;
        DescriptorSetBuilder& builder;
    };
    friend class DescriptorSetUpdater;

    void addBinding(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stages, vk::Sampler* pImmutableSampler = nullptr);
    void setMaximumSetCount(uint32_t count);

    void createLayout();
    void destroyLayout();

    vk::DescriptorSet allocateSet();
    std::vector<vk::DescriptorSet> allocateSets(uint32_t count);
    template<size_t C>
    std::array<vk::DescriptorSet, C> allocateSets()
    {
        assert(mPoolBuilder);
        std::array<vk::DescriptorSet, C> sets;
        std::array<vk::DescriptorSetLayout, C> layouts;
        layouts.fill(mLayout);
        vk::DescriptorSetAllocateInfo allocInfo = {};
        allocInfo.descriptorPool = mPoolBuilder->mPool;
        allocInfo.pSetLayouts = layouts.data();
        allocInfo.descriptorSetCount = C;
        VK_CHECK_RESULT(theVulkanLayer.device.allocateDescriptorSets(&allocInfo, sets.data()))
        return sets;
    }
    void freeSets(vk::ArrayProxy<vk::DescriptorSet> sets);

    vk::DescriptorSetLayout layout() const;
    DescriptorSetUpdater update(vk::ArrayProxy<vk::DescriptorSet> descriptorSets);

private:
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    std::vector<vk::DescriptorBindingFlagsEXT> extFlags;
    vk::DescriptorSetLayout mLayout = nullptr;
    DescriptorPoolBuilder* mPoolBuilder;
    uint32_t mMaxSetCount;
};

struct PipelineBuildHelpers {
    static const PipelineBuildHelpers& instance() {
        static PipelineBuildHelpers inst = {};
        return inst;
    }

    PipelineBuildHelpers();

    vk::PipelineVertexInputStateCreateInfo emptyVertexInputState;
    vk::PipelineInputAssemblyStateCreateInfo triangleListInputAssembly;
    vk::PipelineViewportStateCreateInfo dynamicViewportScissorState;
    vk::PipelineDepthStencilStateCreateInfo disabledDepthStencilTest;
    vk::PipelineMultisampleStateCreateInfo disabledMultisampling;

    template <vk::DynamicState D1, vk::DynamicState... DRest>
    vk::PipelineDynamicStateCreateInfo* getDynamicState() const
    {
        static auto dynamicStates = std::array<vk::DynamicState, sizeof...(DRest) + 1>{D1, DRest...};
        static vk::PipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.pDynamicStates = dynamicStates.data();
        dynamicState.dynamicStateCount = dynamicStates.size();
        return &dynamicState;
    }

    template <uint8_t N>
    vk::PipelineColorBlendStateCreateInfo* getDisabledColorBlendingState() const
    {
        static std::array<vk::PipelineColorBlendAttachmentState, N> colorBlendAttachments = {};
        for (auto& it : colorBlendAttachments) {
            it.blendEnable = VK_FALSE;
            it.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        }

        static vk::PipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = colorBlendAttachments.size();
        colorBlending.pAttachments = colorBlendAttachments.data();

        return &colorBlending;
    }

    vk::Pipeline createFullscreenPipeline(vk::PipelineLayout layout, vk::RenderPass renderPass, uint32_t subpass, vk::PipelineShaderStageCreateInfo fragmentShader, std::optional<vk::PipelineShaderStageCreateInfo> vertexShader = std::nullopt) const;
    vk::Pipeline createFullscreenPipeline(vk::PipelineLayout layout, vk::PipelineRenderingCreateInfo dynamicRenderingInfo, vk::PipelineShaderStageCreateInfo fragmentShader, std::optional<vk::PipelineShaderStageCreateInfo> vertexShader = std::nullopt) const;

private:
    vk::Viewport dynamicViewport;
    vk::Rect2D dynamicScissor;
};

inline const auto& thePipelineBuildHelpers = PipelineBuildHelpers::instance();

template <typename T>
class DependentRendererComponentContainer
{
public:
    template <typename F>
    T& initializeElement(size_t i, F&& initializer)
    {
        if (i >= components.size()) components.resize(i + 1, std::nullopt);
        if (!components[i]) {
            components[i] = T{};
            initializer(components[i].value());
        }
        return components[i].value();
    }

    void destroy()
    {
        for (auto& it : components) {
            if (it) it->destroy();
        }
        components.clear();
    }

private:
    std::deque<std::optional<T>> components;
};

class VertexInputBindingBuilder {
public:
    VertexInputBindingBuilder& addAttribute(vk::VertexInputRate inputRate, uint32_t location, vk::Format format, uint32_t stride, uint32_t offset = 0);
    VertexInputBindingBuilder& addInstanceAttribute(uint32_t location, vk::Format format, uint32_t stride, uint32_t offset = 0);
    VertexInputBindingBuilder& addVertexAttribute(uint32_t location, vk::Format format, uint32_t stride, uint32_t offset = 0);
    vk::PipelineVertexInputStateCreateInfo build();

private:
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attribs;
};

#endif//VULKAN_INTRO_COMMON_RENDERING_H
