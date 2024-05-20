#ifndef VULKAN_INTRO_WINDOW_MANAGER_H
#define VULKAN_INTRO_WINDOW_MANAGER_H

struct GLFWwindow;

class WindowManager
{
public:
    static WindowManager& instance() { static WindowManager wm; return wm; }

    void initializeWindow();
    void prepareSwapChain();
    void registerCallbacks();
    bool initializeSwapChain();

    void destroyWindow();
    void destroySwapChain();

    void updateGui();

    void close();
    void invalidate();
    bool shouldClose();
    bool ready() const;
    bool stable() const;
    glm::ivec2 getResolution() const;
    void setResolution(glm::ivec2 resolution);

    bool isVisible() const;
    bool isFullscreen() const;

    void toggleFullscreen();

    void backupCurrentContext();
    void restoreCurrentContext();

private:
    WindowManager() : window{ nullptr } {}
    GLFWwindow* createWindow();
    void createSurface();
    bool createSwapChain();
    void createImageViews();

public:
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };
    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& dev);

private:
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

public:
    GLFWwindow* window, *backup;
    vk::SurfaceKHR surface;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::SwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent, nextSwapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews, swapChainImageViewsUnorm;
    bool frameBufferResized = false;
    SwapChainSupportDetails swapChainSupportDetails;
    glm::ivec2 currentPosition;

private:
    bool fullscreen;
    glm::ivec2 lastPosition, lastResolution;
};

inline auto& theWindowManager = WindowManager::instance();

#endif//VULKAN_INTRO_WINDOW_MANAGER_H
