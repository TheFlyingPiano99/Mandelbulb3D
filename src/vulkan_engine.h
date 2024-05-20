#ifndef VULKAN_INTRO_VULKAN_ENGINE_H
#define VULKAN_INTRO_VULKAN_ENGINE_H

class Renderer;
class PerspectiveCamera;
class GameScene;

class VulkanEngine {
public:
    enum class Result {
        Success, SwapChainOutOfDate
    };

public:
    static VulkanEngine& instance() { static VulkanEngine ve; return ve; }
    void initialize();
    void destroy();
    void updateGui();
    Result drawFrame(Renderer* pRenderer, PerspectiveCamera* pCamera, GameScene* pGameScene);
    void waitIdle();

private:
    VulkanEngine() = default;
    void createCommandBuffers();
    void createSemaphores();

private:
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;

    size_t currentFrame = 0;
};

inline auto& theVulkanEngine = VulkanEngine::instance();

#endif//VULKAN_INTRO_VULKAN_ENGINE_H
