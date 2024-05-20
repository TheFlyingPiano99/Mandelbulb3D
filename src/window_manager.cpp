#include "window_manager.h"
#include "vulkan_layer.h"
#include <GLFW/glfw3.h>

void WindowManager::initializeWindow()
{
    window = createWindow();
    createSurface();
}

void WindowManager::registerCallbacks()
{
    theInputManager.registerWindowResizeHandler([&](int x, int y) {
        frameBufferResized = true;
        nextSwapChainExtent.width = (uint32_t)x;
        nextSwapChainExtent.height = (uint32_t)y;
    });

    theInputManager.registerWindowMoveHandler([&](int x, int y) {
        currentPosition.x = x;
        currentPosition.y = y;
    });
}

void WindowManager::prepareSwapChain()
{
    surfaceFormat = chooseSwapSurfaceFormat(theVulkanLayer.physicalDevice.getSurfaceFormatsKHR(surface));
    swapChainImageFormat = surfaceFormat.format;
}

bool WindowManager::initializeSwapChain()
{
    auto ret = createSwapChain();
    if (!ret) return false;
    createImageViews();
    frameBufferResized = false;
    return true;
}

void WindowManager::destroyWindow()
{
    theVulkanLayer.instance.destroySurfaceKHR(surface);
    glfwDestroyWindow(window);
}

void WindowManager::destroySwapChain()
{
    auto& vl = theVulkanLayer;
    for (auto& swapChainImageView : swapChainImageViews) {
        vl.device.destroy(swapChainImageView);
    }
    swapChainImageViews.clear();

    for (auto& swapChainImageView : swapChainImageViewsUnorm) {
        vl.device.destroy(swapChainImageView);
    }
    swapChainImageViewsUnorm.clear();

    vl.device.destroy(swapChain);
}

void WindowManager::updateGui()
{
    ImGui::Begin("MainWindow", nullptr, 
        ImGuiWindowFlags_NoBackground | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoCollapse | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoMouseInputs

    );
    ImGui::SetWindowSize(ImVec2{ (float)swapChainExtent.width, (float)swapChainExtent.height });
    ImGui::SetWindowPos(ImVec2{ (float)currentPosition.x, (float)currentPosition.y });

    if (ImGui::BeginPopupContextWindow()) {
        ImGui::Text("teszt");
        ImGui::EndPopup();
    }

    ImGui::End();
}

void WindowManager::close()
{
    glfwSetWindowShouldClose(window, 1);
}

void WindowManager::invalidate()
{
    frameBufferResized = true;
}

bool WindowManager::shouldClose()
{
    return glfwWindowShouldClose(window) != 0;
}

bool WindowManager::ready() const
{
    return !frameBufferResized;
}

bool WindowManager::stable() const
{
    return  glfwGetTime() - theInputManager.lastResizeTime.load(std::memory_order::relaxed) > 0.1 &&
            nextSwapChainExtent.height > 0 && nextSwapChainExtent.width > 0;
}

glm::ivec2 WindowManager::getResolution() const
{
    return {swapChainExtent.width, swapChainExtent.height};
}

void WindowManager::setResolution(glm::ivec2 resolution)
{
    theInputManager.executeOn([w = window, res = resolution]() {
        glfwSetWindowSize(w, res.x, res.y);
    });
}

bool WindowManager::isVisible() const
{
    return glm::any(glm::greaterThan(getResolution(), {0, 0}));
}

bool WindowManager::isFullscreen() const
{
    return fullscreen;
}

void WindowManager::toggleFullscreen()
{
    fullscreen = !fullscreen;
    if (fullscreen) {
        lastPosition = currentPosition;
        lastResolution = { swapChainExtent.width, swapChainExtent.height };
    }

    theInputManager.executeOn([w = window, fs = fullscreen, lp = lastPosition, lr = lastResolution]() {
        if (fs) {
            auto primaryMonitor = glfwGetPrimaryMonitor();
            auto videoMode = glfwGetVideoMode(primaryMonitor);
            glfwSetWindowAttrib(w, GLFW_DECORATED, GLFW_FALSE);
            glfwSetWindowMonitor(w, nullptr, 0, 0, videoMode->width, videoMode->height, videoMode->refreshRate);
        } else {
            glfwSetWindowAttrib(w, GLFW_DECORATED, GLFW_TRUE);
            glfwSetWindowMonitor(w, nullptr, lp.x, lp.y, lr.x, lr.y, 0);
        }
    });
}

void WindowManager::backupCurrentContext()
{
    backup = glfwGetCurrentContext();
}

void WindowManager::restoreCurrentContext()
{
    glfwMakeContextCurrent(backup);
}

GLFWwindow* WindowManager::createWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto pWindow = glfwCreateWindow(800, 600, "VulkanIntro", nullptr, nullptr);
    fullscreen = false;
    glfwGetWindowPos(pWindow, &currentPosition.x, &currentPosition.y);
    return pWindow;
}

void WindowManager::createSurface()
{
    VkSurfaceKHR ret;
    if (glfwCreateWindowSurface((VkInstance)theVulkanLayer.instance, window, nullptr, &ret) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::SurfaceKHR{ret};
}

bool WindowManager::createSwapChain()
{
    auto& vl = theVulkanLayer;
    
    swapChainSupportDetails = querySwapChainSupport(vl.physicalDevice);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupportDetails.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupportDetails.capabilities);

    if (extent.width == 0 || extent.height == 0) return false;

    uint32_t imageCount = swapChainSupportDetails.capabilities.minImageCount + 1;
    if (swapChainSupportDetails.capabilities.maxImageCount > 0 && imageCount > swapChainSupportDetails.capabilities.maxImageCount) {
        imageCount = swapChainSupportDetails.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo = {};
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;

    auto& indices = vl.queueFamilyIndices;
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    createInfo.preTransform = swapChainSupportDetails.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = nullptr;

    vk::ImageFormatListCreateInfo formatList = {};
    std::array<vk::Format, 2> formats = {
        createInfo.imageFormat, vk::Format::eB8G8R8A8Unorm
    };
    if (vl.swapChainMutableFormatAvailable) {
        formatList.viewFormatCount = formats.size();
        formatList.pViewFormats = formats.data();

        createInfo.pNext = &formatList;
        createInfo.flags = vk::SwapchainCreateFlagBitsKHR::eMutableFormat;
    }

    swapChain = vl.device.createSwapchainKHR(createInfo);
    swapChainImages = vl.device.getSwapchainImagesKHR(swapChain);

    swapChainExtent = extent;

    return true;
}

void WindowManager::createImageViews()
{
    swapChainImageViews.reserve(swapChainImages.size());
    for (auto& swapChainImage : swapChainImages) {
        swapChainImageViews.push_back(theVulkanLayer.createImageView(swapChainImage, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1));
    }

    if (theVulkanLayer.swapChainMutableFormatAvailable) {
        swapChainImageViewsUnorm.reserve(swapChainImages.size());
        for (auto& swapChainImage : swapChainImages) {
            swapChainImageViewsUnorm.push_back(theVulkanLayer.createImageView(swapChainImage, vk::Format::eB8G8R8A8Unorm, vk::ImageAspectFlagBits::eColor, 1));
        }
    } else {
        swapChainImageViewsUnorm = swapChainImageViews;
    }
}

WindowManager::SwapChainSupportDetails WindowManager::querySwapChainSupport(const vk::PhysicalDevice& dev)
{
    SwapChainSupportDetails details;
    details.capabilities = dev.getSurfaceCapabilitiesKHR(surface);
    details.formats = dev.getSurfaceFormatsKHR(surface);
    details.presentModes = dev.getSurfacePresentModesKHR(surface);
    return details;
}

vk::SurfaceFormatKHR WindowManager::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
        return {vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR WindowManager::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
            return availablePresentMode;
        } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

vk::Extent2D WindowManager::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
        };
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}
