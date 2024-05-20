#include "vulkan_engine.h"
#include "vulkan_layer.h"
#include "window_manager.h"
#include "gui/gui_manager.h"
#include "shader_manager.h"
#include "rendering/renderer.h"

namespace fs = std::filesystem;
using namespace glfwim;

void VulkanEngine::initialize()
{
    createCommandBuffers();
    createSemaphores();
}

void VulkanEngine::destroy()
{
    auto& vl = theVulkanLayer;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vl.device.destroy(renderFinishedSemaphores[i]);
        vl.device.destroy(imageAvailableSemaphores[i]);
        vl.device.destroy(inFlightFences[i]);
    }
    renderFinishedSemaphores.clear();
    imageAvailableSemaphores.clear();
    inFlightFences.clear();

    vl.device.freeCommandBuffers(vl.commandPool, commandBuffers);
}

void VulkanEngine::updateGui()
{
    if (ImGui::Begin("Settings")) {
        if (ImGui::Button("Reload shaders")) {
            waitIdle();
            theShaderManager.rebuildShaders();
        }

        ImGui::SameLine();
        if (ImGui::Button("Reload all shaders")) {
            waitIdle();
            theShaderManager.rebuildShaders(true);
        }
    }

    ImGui::End();
}

void VulkanEngine::waitIdle()
{
    theVulkanLayer.safeWaitIdle();
}

VulkanEngine::Result VulkanEngine::drawFrame(Renderer* pRenderer, PerspectiveCamera* pCamera, GameScene* pGameScene)
{
    auto& vl = theVulkanLayer;

    if (theWindowManager.swapChainExtent.width == 0 || theWindowManager.swapChainExtent.height == 0) {
        return Result::SwapChainOutOfDate;
    }

    VK_CHECK_RESULT(vl.device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max()))

    uint32_t imageIndex;
    auto result = vl.device.acquireNextImageKHR(
            theWindowManager.swapChain, std::numeric_limits<uint64_t>::max(),
            imageAvailableSemaphores[currentFrame], nullptr, &imageIndex
    );

    if (result == vk::Result::eErrorOutOfDateKHR) {
        return Result::SwapChainOutOfDate;
    } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vk::CommandBufferBeginInfo beginInfo = {};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffers[currentFrame].begin(beginInfo);

    vk::ImageMemoryBarrier2 bar = {};
    bar.image = theWindowManager.swapChainImages[imageIndex];
    bar.oldLayout = vk::ImageLayout::eUndefined;
    bar.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    bar.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    bar.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    bar.dstAccessMask = vk::AccessFlagBits2KHR::eColorAttachmentWrite | vk::AccessFlagBits2KHR::eColorAttachmentRead;
    bar.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    bar.subresourceRange.baseArrayLayer = 0;
    bar.subresourceRange.layerCount = 1;
    bar.subresourceRange.baseMipLevel = 0;
    bar.subresourceRange.levelCount = 1;

    vk::DependencyInfo dep = {};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &bar;

    commandBuffers[currentFrame].pipelineBarrier2(dep);

    ImGui::Begin("Renderer parameters");
    auto rctx = RenderContext{
        pGameScene, pCamera,
        commandBuffers[currentFrame],
        currentFrame, imageIndex
    };

    pRenderer->render(rctx);

    ImGui::End();

    theGUIManager.render(RenderContext {
            nullptr, nullptr,
            commandBuffers[currentFrame],
            currentFrame, imageIndex
    });

    commandBuffers[currentFrame].end();

    vk::SubmitInfo submitInfo = {};

    vk::Semaphore waitSemaphores[] = {pRenderer->vulkanWaitsForCudaSemaphore, imageAvailableSemaphores[currentFrame]};
    vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eAllCommands};
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame], pRenderer->cudaWaitsForVulkanSemaphore };
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VK_CHECK_RESULT(vl.device.resetFences(1, &inFlightFences[currentFrame]))

    auto res = vl.safeSubmitGraphicsCommand(submitInfo, false, inFlightFences[currentFrame]);
    if (res != vk::Result::eSuccess) {
        vulkanLogger.LogError("Failed to submit draw command buffer!");
    }

    vk::PresentInfoKHR presentInfo = {};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    vk::SwapchainKHR swapChains[] = {theWindowManager.swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // Optional

    Result ret = Result::Success;
    result = vk::Result::eErrorOutOfDateKHR;
    try {
        result = vl.safeSubmitPresentCommand(presentInfo);
    } catch (...){}

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || theWindowManager.frameBufferResized) {
        ret = Result::SwapChainOutOfDate;
    } else if (result != vk::Result::eSuccess) {
        vulkanLogger.LogError("Failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1ul) % MAX_FRAMES_IN_FLIGHT;
    return ret;
}

void VulkanEngine::createCommandBuffers()
{
    auto& vl = theVulkanLayer;
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = vl.commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    commandBuffers = vl.device.allocateCommandBuffers(allocInfo);
}

void VulkanEngine::createSemaphores()
{
    auto& vl = theVulkanLayer;
    imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo = {};

    vk::FenceCreateInfo fenceInfo = {};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        imageAvailableSemaphores.push_back(vl.device.createSemaphore(semaphoreInfo));
        renderFinishedSemaphores.push_back(vl.device.createSemaphore(semaphoreInfo));
        inFlightFences.push_back(vl.device.createFence(fenceInfo));
    }
}
