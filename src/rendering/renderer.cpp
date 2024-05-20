#include "rendering/renderer.h"
#include "shader_manager.h"
#include "rendering/common_rendering.h"
#include "window_manager.h"
#include "vulkan_layer.h"
#include "gui/gui_manager.h"
#include "scene/game_object.h"
#include "scene/lights.h"
#include "scene/game_scene.h"
#include "asset_managment/asset_manager.h"
#include "camera/perspective_camera.h"
#include "utility/utility.hpp"
#include "rendering/renderer_dependency_provider.h"
#include "cuda/interop.cuh"
#include "dxgi1_2.h"

void Renderer::update(double dt)
{
    /* TODO: pan and zoom */
}

void Renderer::initialize()
{
    query.create();
    createResources();
    createPipelineLayout();
    createFsPipeline();
    createSyncObjects();
}

void Renderer::destroy()
{
    query.destroy();
    descriptorPoolBuilder.destroy();
    perFrameDscSetBuilder.destroyLayout();
    for (auto& it : perFrameBuffers) it.destroy();
    theVulkanLayer.device.destroy(fsSampler);
    theVulkanLayer.device.destroy(fsPipelineLayout);
    theVulkanLayer.device.destroy(fsPipeline);
    vmaDestroyPool(theVulkanLayer.allocator, externalPool);

    freeExportedSemaphores();
    theVulkanLayer.device.destroy(cudaWaitsForVulkanSemaphore);
    theVulkanLayer.device.destroy(vulkanWaitsForCudaSemaphore);
}

void Renderer::initializeSwapChainDependentComponents()
{
    createImages();
    updateDescriptorSets();
}

void Renderer::destroySwapChainDependentComponents()
{
    freeExportedVulkanImage();
    colorImage.destroy();
}

void Renderer::render(const RenderContext& ctx)
{
    auto& vl = theVulkanLayer;
    auto& wm = theWindowManager;

    // timestamp query
    query.query();
    // not representative with cuda!
    theGUIManager.addStatistic("Renderer", std::make_tuple("Frame time: ", (query.results[1] - query.results[0]) * vl.timestampPeriod / 1e6, " ms"));
    
    auto vp = vk::Viewport{ 0, 0, (float)wm.swapChainExtent.width, (float)wm.swapChainExtent.height, 0.0, 1.0 };

    // Update uniforms
    // per frame uniforms /* unused */
    PerFrameUniformData fud = {}; // prepare uniform data on CPU
    fud.v = ctx.cam->V();
    fud.ambientLight = glm::vec4{ ctx.pGameScene->ambientLight.color, ctx.pGameScene->ambientLight.power };
    memcpy(perFrameBuffers[ctx.frameID].allocationInfo.pMappedData, &fud, sizeof(PerFrameUniformData)); // upload to GPU through mapped pointer
    perFrameBuffers[ctx.frameID].flush(); // ensure visibility

    query.beginFrame(ctx.cmd); // reset query
    query.write(ctx.cmd, vk::PipelineStageFlagBits::eTopOfPipe); // write timestamp

    // prepare render pass
    vk::RenderingAttachmentInfo colorInfo = {}; // color attachment (render target)
    colorInfo.imageView = wm.swapChainImageViews[ctx.imageID];
    colorInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorInfo.loadOp = vk::AttachmentLoadOp::eClear;
    colorInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorInfo.clearValue.color = vk::ClearColorValue{ std::array<float, 4>{0, 0, 0, 1.0} };

    vk::RenderingInfo renderingInfo = {};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorInfo;
    renderingInfo.renderArea.offset.x = renderingInfo.renderArea.offset.y = 0;
    renderingInfo.renderArea.extent = wm.swapChainExtent;
    renderingInfo.layerCount = 1;    
    
    ctx.cmd.beginRendering(renderingInfo); // start render pass

    // not optimal!! should use external synchronization with semaphores. Must be done in vulkan_engine.cpp at submission

    renderCuda();

    ctx.cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, fsPipelineLayout, 0, 1, &perFrameDescriptorSets[ctx.frameID], 0, nullptr);
    ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, fsPipeline);
    ctx.cmd.setViewport(0, 1, &vp);
    ctx.cmd.setScissor(0, 1, &renderingInfo.renderArea);
    ctx.cmd.draw(3, 1, 0, 0);

    ctx.cmd.endRendering();

    query.write(ctx.cmd, vk::PipelineStageFlagBits::eBottomOfPipe);

    query.endFrame();
}

void Renderer::createPipelineLayout()
{
    descriptorPoolBuilder.initialize();

    perFrameDscSetBuilder.setMaximumSetCount(MAX_FRAMES_IN_FLIGHT);
    perFrameDscSetBuilder.addBinding(0, vk::DescriptorType::eUniformBuffer, // perFrameUniformData
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
    );
    perFrameDscSetBuilder.addBinding(1, vk::DescriptorType::eCombinedImageSampler, // cuda image
        vk::ShaderStageFlagBits::eFragment, &fsSampler
    );
    perFrameDscSetBuilder.createLayout();

    descriptorPoolBuilder.create();

    perFrameDescriptorSets = perFrameDscSetBuilder.allocateSets<MAX_FRAMES_IN_FLIGHT>();

    fsPipelineLayout = theVulkanLayer.name(theVulkanLayer.createPipelineLayout(
        { perFrameDscSetBuilder.layout() }
    ), "Renderer_FSPipelineLayout");
}

void Renderer::updateDescriptorSets()
{
    perFrameDscSetBuilder.update(perFrameDescriptorSets)
        .writeBuffer(0, perFrameBuffers)
        .writeImage(1, colorImage.view, vk::ImageLayout::eGeneral);
}

void Renderer::createImages()
{
    auto& vl = theVulkanLayer;
    auto& wm = theWindowManager;

    // we are using the custom pool
    VmaAllocationCreateInfo alloc = {};
    alloc.usage = (VmaMemoryUsage)vma::MemoryUsage::eAutoPreferDevice;
    alloc.pool = externalPool;

    // and we want to export the image
    vk::ExternalMemoryImageCreateInfo extInfo = {};
    extInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

    colorImage = vl.name(vl.createImage(wm.swapChainExtent.width, wm.swapChainExtent.height, 1,
        vk::SampleCountFlagBits::e1,
        vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eGeneral, // general layout so it is compatible (doesnt really matter as CUDA is nvidia specific)
        vk::ImageAspectFlagBits::eColor,
        vk::ImageUsageFlagBits::eSampled, alloc, vk::ImageTiling::eOptimal, &extInfo // extension struct passed here
    ), "Renderer_ColorImage");

    colorImageNativeHandle = colorImage.getNativeWin32Handle(); // receive a native win32 handle
    // and finally export the image to CUDA
    exportVulkanImageToCuda_R8G8B8A8Unorm(colorImageNativeHandle, colorImage.allocationInfo.size, colorImage.allocationInfo.offset, colorImage.width, colorImage.height);
}

void Renderer::createResources()
{
    auto& vl = theVulkanLayer;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        perFrameBuffers[i] = vl.createBuffer(sizeof(PerFrameUniformData), vk::BufferUsageFlagBits::eUniformBuffer, vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit | vma::AllocationCreateFlagBits::eCreateMappedBit);
    }

    vk::SamplerCreateInfo info = {};
    info.magFilter = vk::Filter::eLinear;
    info.minFilter = vk::Filter::eLinear;
    info.mipmapMode = vk::SamplerMipmapMode::eLinear;
    info.addressModeU = vk::SamplerAddressMode::eRepeat;
    info.addressModeV = vk::SamplerAddressMode::eRepeat;
    info.addressModeW = vk::SamplerAddressMode::eRepeat;
    info.minLod = -1000;
    info.maxLod = 1000;
    info.maxAnisotropy = 1.0f;
    fsSampler = vl.device.createSampler(info);


    // we create the vma pool. Most parameters are default
    VmaPoolCreateInfo poolInfo = {};
    poolInfo.blockSize = 0;
    poolInfo.maxBlockCount = 0;
    poolInfo.minBlockCount = 0;
    poolInfo.minAllocationAlignment = 0;
    poolInfo.priority = 1.0;
    poolInfo.flags = {};
    
    // example image similar to those that we will alocate from the pool
    // this way the pool can select a correct memory type
    VkImageCreateInfo example = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    example.format = (VkFormat)vk::Format::eR8G8B8A8Unorm;
    example.usage = (VkImageUsageFlags)vk::ImageUsageFlagBits::eSampled;
    example.imageType = (VkImageType)vk::ImageType::e2D;
    example.initialLayout = (VkImageLayout)vk::ImageLayout::eUndefined;
    example.samples = (VkSampleCountFlagBits)vk::SampleCountFlagBits::e1;
    example.tiling = (VkImageTiling)vk::ImageTiling::eOptimal;
    example.sharingMode = (VkSharingMode)vk::SharingMode::eExclusive;
    example.extent.width = example.extent.height = 100;
    example.extent.depth = 1;
    example.arrayLayers = 1;
    example.mipLevels = 1;
    example.flags = {};
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.requiredFlags = (VkMemoryPropertyFlags)vk::MemoryPropertyFlagBits::eDeviceLocal;
    allocInfo.usage = (VmaMemoryUsage)vma::MemoryUsage::eAutoPreferDevice;
    vmaFindMemoryTypeIndexForImageInfo(vl.allocator, &example, &allocInfo, &poolInfo.memoryTypeIndex); // find the memory type

    // define windows security attributes and settings for the allocations
    vulkanExportMemoryWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vulkanExportMemoryWin32HandleInfoKHR.pNext = nullptr;
    vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vulkanExportMemoryWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)nullptr;

    // allocation can be exported as a native win32 handle
    exportMemoryInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportMemoryInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    exportMemoryInfo.pNext = &vulkanExportMemoryWin32HandleInfoKHR;

    poolInfo.pMemoryAllocateNext = &exportMemoryInfo;
    vmaCreatePool(vl.allocator, &poolInfo, &externalPool); // create the pool
}

void Renderer::createFsPipeline()
{
    // simple fullscreen pipeline to copy a texture to the swap chain image
    fsPipelineRef = theShaderManager.addAndExecuteShaderDependency({ "copy.frag" }, [this](auto shaders) {
        std::vector<vk::PipelineShaderStageCreateInfo> ss; ss.reserve(shaders.size());
        std::transform(shaders.begin(), shaders.end(), std::back_inserter(ss), [&](auto& s) { return s.build(); });

        vk::PipelineRenderingCreateInfo rendering = {};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &theWindowManager.swapChainImageFormat;

        if (fsPipeline) theVulkanLayer.device.destroy(fsPipeline);
        fsPipeline = thePipelineBuildHelpers.createFullscreenPipeline(fsPipelineLayout, rendering, ss[0]);
        theShaderManager.destroyShaders(ss);
    });
}

void Renderer::createSyncObjects()
{
    vk::SemaphoreCreateInfo semaphoreInfo;

    WindowsSecurityAttributes winSecurityAttributes;

    vk::ExportSemaphoreWin32HandleInfoKHR vulkanExportSemaphoreWin32HandleInfoKHR = {};
    vulkanExportSemaphoreWin32HandleInfoKHR.pNext = nullptr;
    vulkanExportSemaphoreWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vulkanExportSemaphoreWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vulkanExportSemaphoreWin32HandleInfoKHR.name = (LPCWSTR)NULL;

    vk::ExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
    vulkanExportSemaphoreCreateInfo.pNext = &vulkanExportSemaphoreWin32HandleInfoKHR;
    vulkanExportSemaphoreCreateInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBitsKHR::eOpaqueWin32;

    semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

    cudaWaitsForVulkanSemaphore = theVulkanLayer.device.createSemaphore(semaphoreInfo);
    vulkanWaitsForCudaSemaphore = theVulkanLayer.device.createSemaphore(semaphoreInfo);

    exportSemaphoresToCuda(theVulkanLayer.getNativeWin32Handle(cudaWaitsForVulkanSemaphore), theVulkanLayer.getNativeWin32Handle(vulkanWaitsForCudaSemaphore));
}