#include "common_rendering.h"
#include "shader_manager.h"

void DescriptorPoolBuilder::initialize(vk::DescriptorPoolCreateFlags flags)
{
	setCount = 0;
	mPool = nullptr;
	descriptorCounts.clear();
	mFlags = flags;
}

void DescriptorPoolBuilder::create()
{
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (auto& it : descriptorCounts) {
        vk::DescriptorPoolSize s;
        s.type = it.first;
        s.descriptorCount = it.second;
        poolSizes.push_back(s);
    }

    vk::DescriptorPoolCreateInfo poolInfo = {};
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = setCount;
    poolInfo.flags = mFlags;
    mPool = theVulkanLayer.device.createDescriptorPool(poolInfo);
}

void DescriptorPoolBuilder::destroy()
{
    theVulkanLayer.device.destroy(mPool);
}

void DescriptorPoolBuilder::reset()
{
	theVulkanLayer.device.resetDescriptorPool(mPool);
}

void DescriptorSetBuilder::addBinding(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stages, vk::Sampler* pImmutableSampler)
{
    vk::DescriptorSetLayoutBinding layoutBinding;
    layoutBinding.binding = binding;
    layoutBinding.descriptorCount = 1;
    layoutBinding.descriptorType = type;
    layoutBinding.stageFlags = stages;
    layoutBinding.pImmutableSamplers = pImmutableSampler;

    vk::DescriptorBindingFlagsEXT flags = {};

    bindings.push_back(layoutBinding);
    extFlags.push_back(flags);
}

void DescriptorSetBuilder::setMaximumSetCount(uint32_t count)
{
    mMaxSetCount = count;
}

void DescriptorSetBuilder::createLayout()
{
    vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT extInfo = {};
    extInfo.pBindingFlags = extFlags.data();
    extInfo.bindingCount = (uint32_t)extFlags.size();

    vk::DescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    layoutInfo.pNext = &extInfo;
    if (!mPoolBuilder) { // no pool so it must be a push descriptor
        layoutInfo.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
    }
    
    mLayout = theVulkanLayer.device.createDescriptorSetLayout(layoutInfo);

    if (mPoolBuilder) {
		mPoolBuilder->setCount += mMaxSetCount;
		for (auto& it : bindings) {
			mPoolBuilder->descriptorCounts[it.descriptorType] += it.descriptorCount * mMaxSetCount;
		}
    }
}

void DescriptorSetBuilder::destroyLayout()
{
    theVulkanLayer.device.destroy(mLayout);
    mMaxSetCount = 0;
    bindings.clear();
    extFlags.clear();
}

vk::DescriptorSet DescriptorSetBuilder::allocateSet()
{
    assert(mPoolBuilder);
    vk::DescriptorSet set;
    vk::DescriptorSetAllocateInfo allocInfo = {};
    allocInfo.descriptorPool = mPoolBuilder->mPool;
    allocInfo.pSetLayouts = &mLayout;
    allocInfo.descriptorSetCount = 1;
    VK_CHECK_RESULT(theVulkanLayer.device.allocateDescriptorSets(&allocInfo, &set))
    return set;
}

std::vector<vk::DescriptorSet> DescriptorSetBuilder::allocateSets(uint32_t count)
{
    assert(mPoolBuilder);
    std::vector<vk::DescriptorSet> sets;
    sets.resize(count);
    std::vector<vk::DescriptorSetLayout> layouts = {};
    layouts.resize(count, mLayout);
    vk::DescriptorSetAllocateInfo allocInfo = {};
    allocInfo.descriptorPool = mPoolBuilder->mPool;
    allocInfo.pSetLayouts = layouts.data();
    allocInfo.descriptorSetCount = (uint32_t)layouts.size();
    VK_CHECK_RESULT(theVulkanLayer.device.allocateDescriptorSets(&allocInfo, sets.data()))
    return sets;
}

void DescriptorSetBuilder::freeSets(vk::ArrayProxy<vk::DescriptorSet> sets)
{
    assert(mPoolBuilder);
    vk::ArrayProxy<const vk::DescriptorSet> temp{ sets.size(), sets.data() };
    theVulkanLayer.device.freeDescriptorSets(mPoolBuilder->mPool, temp);
}

vk::DescriptorSetLayout DescriptorSetBuilder::layout() const
{
    return mLayout;
}

DescriptorSetBuilder::DescriptorSetUpdater DescriptorSetBuilder::update(vk::ArrayProxy<vk::DescriptorSet> descriptorSets)
{
    return DescriptorSetUpdater{ *this, descriptorSets };
}

DescriptorSetBuilder::DescriptorSetUpdater::DescriptorSetUpdater(DescriptorSetBuilder& descSetBuilder,  vk::ArrayProxy<vk::DescriptorSet> descriptorSets)
    : builder{ descSetBuilder }
{
    descSets = descriptorSets;
    descWrites.resize(descSets.size());
    imageInfos.resize(descSets.size());
    bufferInfos.resize(descSets.size());
    tlasInfos.resize(descSets.size());
}

DescriptorSetBuilder::DescriptorSetUpdater& DescriptorSetBuilder::DescriptorSetUpdater::writeBuffer(uint32_t binding, vk::ArrayProxy<VulkanBuffer> buffers, vk::DeviceSize offset, vk::DeviceSize range)
{
    std::vector<vk::Buffer> bfs {};
    bfs.reserve(buffers.size());
    for (auto& it : buffers) {
        bfs.push_back(it.buffer);
    }
    return writeBuffer(binding, bfs, offset, range);
}

DescriptorSetBuilder::DescriptorSetUpdater& DescriptorSetBuilder::DescriptorSetUpdater::writeBuffer(uint32_t binding, vk::ArrayProxy<vk::Buffer> buffers, vk::DeviceSize offset, vk::DeviceSize range)
{
    assert(buffers.size() == 1 || buffers.size() == descWrites.size());
    auto found = std::find_if(builder.bindings.begin(), builder.bindings.end(), [&](const auto& it) { return it.binding == binding; });
    assert(found != builder.bindings.end());
   
    vk::DescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = buffers.front();
    bufferInfo.offset = offset;
    bufferInfo.range = range;
    if (buffers.size() == 1) {
        std::fill(bufferInfos.begin(), bufferInfos.end(), bufferInfo);
    } else {
        for (size_t i = 0; i < bufferInfos.size(); ++i) {
            bufferInfo.buffer = *(buffers.begin() + i);
            bufferInfos[i] = bufferInfo;
        }
    }

    for (size_t i = 0; i < descWrites.size(); ++i) {
        descWrites[i].dstSet = *(descSets.begin() + i);
        descWrites[i].dstBinding = binding;
        descWrites[i].dstArrayElement = 0;
        descWrites[i].descriptorType = found->descriptorType;
        descWrites[i].descriptorCount = 1;
        descWrites[i].pBufferInfo = &bufferInfos[i];
    }

    theVulkanLayer.device.updateDescriptorSets(descWrites, nullptr);
    return *this;
}

DescriptorSetBuilder::DescriptorSetUpdater& DescriptorSetBuilder::DescriptorSetUpdater::writeImage(uint32_t binding, vk::ArrayProxy<vk::ImageView> images, vk::ImageLayout imageLayout)
{
    assert(images.size() == 1 || images.size() == descWrites.size());
    auto found = std::find_if(builder.bindings.begin(), builder.bindings.end(), [&](const auto& it) { return it.binding == binding; });
    assert(found != builder.bindings.end());

    vk::DescriptorImageInfo imageInfo = {};
    imageInfo.imageView = images.front();
    imageInfo.imageLayout = imageLayout;
    if (images.size() == 1) {
        std::fill(imageInfos.begin(), imageInfos.end(), imageInfo);
    }
    else {
        for (size_t i = 0; i < bufferInfos.size(); ++i) {
            imageInfo.imageView = *(images.begin() + i);
            imageInfos[i] = imageInfo;
        }
    }

    for (size_t i = 0; i < descWrites.size(); ++i) {
        descWrites[i].dstSet = *(descSets.begin() + i);
        descWrites[i].dstBinding = binding;
        descWrites[i].dstArrayElement = 0;
        descWrites[i].descriptorType = found->descriptorType;
        descWrites[i].descriptorCount = 1;
        descWrites[i].pImageInfo = &imageInfos[i];
    }

    theVulkanLayer.device.updateDescriptorSets(descWrites, nullptr);
    return *this;
}

DescriptorSetBuilder::DescriptorSetUpdater& DescriptorSetBuilder::DescriptorSetUpdater::writeSampler(uint32_t binding, vk::ArrayProxy<vk::Sampler> samplers)
{
    assert(samplers.size() == 1 || samplers.size() == descWrites.size());
    auto found = std::find_if(builder.bindings.begin(), builder.bindings.end(), [&](const auto& it) { return it.binding == binding; });
    assert(found != builder.bindings.end());

    vk::DescriptorImageInfo imageInfo = {};
    imageInfo.sampler = samplers.front();
    if (samplers.size() == 1) {
        std::fill(imageInfos.begin(), imageInfos.end(), imageInfo);
    }
    else {
        for (size_t i = 0; i < bufferInfos.size(); ++i) {
            imageInfo.sampler = *(samplers.begin() + i);
            imageInfos[i] = imageInfo;
        }
    }

    for (size_t i = 0; i < descWrites.size(); ++i) {
        descWrites[i].dstSet = *(descSets.begin() + i);
        descWrites[i].dstBinding = binding;
        descWrites[i].dstArrayElement = 0;
        descWrites[i].descriptorType = found->descriptorType;
        descWrites[i].descriptorCount = 1;
        descWrites[i].pImageInfo = &imageInfos[i];
    }

    theVulkanLayer.device.updateDescriptorSets(descWrites, nullptr);
    return *this;
}


PipelineBuildHelpers::PipelineBuildHelpers()
{
    emptyVertexInputState = vk::PipelineVertexInputStateCreateInfo{};
    emptyVertexInputState.vertexBindingDescriptionCount = 0;
    emptyVertexInputState.pVertexBindingDescriptions = nullptr;
    emptyVertexInputState.vertexAttributeDescriptionCount = 0;
    emptyVertexInputState.pVertexAttributeDescriptions = nullptr;

    triangleListInputAssembly = vk::PipelineInputAssemblyStateCreateInfo{};
    triangleListInputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    triangleListInputAssembly.primitiveRestartEnable = VK_FALSE;

    dynamicViewportScissorState = vk::PipelineViewportStateCreateInfo{};
    dynamicViewportScissorState.viewportCount = 1;
    dynamicViewportScissorState.pViewports = &dynamicViewport;
    dynamicViewportScissorState.scissorCount = 1;
    dynamicViewportScissorState.pScissors = &dynamicScissor;

    disabledDepthStencilTest = vk::PipelineDepthStencilStateCreateInfo{};
    disabledDepthStencilTest.depthTestEnable = VK_FALSE;
    disabledDepthStencilTest.depthWriteEnable = VK_FALSE;
    disabledDepthStencilTest.depthBoundsTestEnable = VK_FALSE;
    disabledDepthStencilTest.stencilTestEnable = VK_FALSE;

    disabledMultisampling = vk::PipelineMultisampleStateCreateInfo{};
    disabledMultisampling.sampleShadingEnable = VK_FALSE;
    disabledMultisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
}

vk::Pipeline PipelineBuildHelpers::createFullscreenPipeline(vk::PipelineLayout layout, vk::RenderPass renderPass, uint32_t subpass, vk::PipelineShaderStageCreateInfo fragmentShader, std::optional<vk::PipelineShaderStageCreateInfo> vertexShader) const
{
    auto shaders = { vertexShader ? vertexShader.value() : theShaderManager.getShader("full_screen_quad.vert").build(), fragmentShader };

    vk::PipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    vk::GraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.stageCount = (uint32_t)shaders.size();
    pipelineInfo.pStages = shaders.begin();
    pipelineInfo.pVertexInputState = &thePipelineBuildHelpers.emptyVertexInputState;
    pipelineInfo.pInputAssemblyState = &thePipelineBuildHelpers.triangleListInputAssembly;
    pipelineInfo.pViewportState = &thePipelineBuildHelpers.dynamicViewportScissorState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &thePipelineBuildHelpers.disabledMultisampling;
    pipelineInfo.pDepthStencilState = &thePipelineBuildHelpers.disabledDepthStencilTest;
    pipelineInfo.pColorBlendState = thePipelineBuildHelpers.getDisabledColorBlendingState<1>();
    pipelineInfo.pDynamicState = thePipelineBuildHelpers.getDynamicState<vk::DynamicState::eViewport, vk::DynamicState::eScissor>();
    pipelineInfo.layout = layout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpass;
    auto ret = theVulkanLayer.createGraphicsPipeline(pipelineInfo);
    if (!vertexShader) theVulkanLayer.device.destroy(shaders.begin()[0].module);
    return ret;
}

vk::Pipeline PipelineBuildHelpers::createFullscreenPipeline(vk::PipelineLayout layout, vk::PipelineRenderingCreateInfo dynamicRenderingInfo, vk::PipelineShaderStageCreateInfo fragmentShader, std::optional<vk::PipelineShaderStageCreateInfo> vertexShader) const
{
	auto shaders = { vertexShader ? vertexShader.value() : theShaderManager.getShader("full_screen_quad.vert").build(), fragmentShader };

	vk::PipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = vk::PolygonMode::eFill;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = vk::CullModeFlagBits::eNone;
	rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
	rasterizer.depthBiasEnable = VK_FALSE;

	vk::GraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.stageCount = (uint32_t)shaders.size();
	pipelineInfo.pStages = shaders.begin();
	pipelineInfo.pVertexInputState = &thePipelineBuildHelpers.emptyVertexInputState;
	pipelineInfo.pInputAssemblyState = &thePipelineBuildHelpers.triangleListInputAssembly;
	pipelineInfo.pViewportState = &thePipelineBuildHelpers.dynamicViewportScissorState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &thePipelineBuildHelpers.disabledMultisampling;
	pipelineInfo.pDepthStencilState = &thePipelineBuildHelpers.disabledDepthStencilTest;
	pipelineInfo.pColorBlendState = thePipelineBuildHelpers.getDisabledColorBlendingState<1>();
	pipelineInfo.pDynamicState = thePipelineBuildHelpers.getDynamicState<vk::DynamicState::eViewport, vk::DynamicState::eScissor>();
	pipelineInfo.layout = layout;
	pipelineInfo.renderPass = nullptr;
	pipelineInfo.pNext = &dynamicRenderingInfo;
	auto ret = theVulkanLayer.createGraphicsPipeline(pipelineInfo);
	if (!vertexShader) theVulkanLayer.device.destroy(shaders.begin()[0].module);
	return ret;
}

VertexInputBindingBuilder& VertexInputBindingBuilder::addAttribute(vk::VertexInputRate inputRate, uint32_t location, vk::Format format, uint32_t stride, uint32_t offset)
{
    vk::VertexInputBindingDescription bind = {};
    bind.binding = (uint32_t)bindings.size();
    bind.stride = stride;
    bind.inputRate = inputRate;
    bindings.push_back(bind);

    vk::VertexInputAttributeDescription attrib = {};
    attrib.format = format;
    attrib.offset = offset;
    attrib.location = location;
    attrib.binding = bindings.back().binding;
    attribs.push_back(attrib);
    return *this;
}

VertexInputBindingBuilder& VertexInputBindingBuilder::addInstanceAttribute(uint32_t location, vk::Format format, uint32_t stride, uint32_t offset)
{
    return addAttribute(vk::VertexInputRate::eInstance, location, format, stride, offset);
}

VertexInputBindingBuilder& VertexInputBindingBuilder::addVertexAttribute(uint32_t location, vk::Format format, uint32_t stride, uint32_t offset)
{
    return addAttribute(vk::VertexInputRate::eVertex, location, format, stride, offset);
}

vk::PipelineVertexInputStateCreateInfo VertexInputBindingBuilder::build()
{
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)bindings.size();
    vertexInputInfo.pVertexBindingDescriptions = bindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)attribs.size();
    vertexInputInfo.pVertexAttributeDescriptions = attribs.data();
    return vertexInputInfo;
}
