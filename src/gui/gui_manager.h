#ifndef VULKAN_INTRO_GUI_MANAGER_H
#define VULKAN_INTRO_GUI_MANAGER_H

class WindowManager;
struct ImGuiIO;
struct ImGuiContext;
struct RenderContext;

class GUIManager {
public:
    static GUIManager& Instance() { static GUIManager instance; return instance; }

    void initialize(WindowManager* pWindowManager);
    void initializeSwapChainDependentComponents();
    void destroy();
    void destroySwapChainDependentComponents();

    void updateGui();
    void startFrame();
    void finishFrame(bool skipRender);
    void render(const RenderContext& ctx);
    bool isMouseCaptured();
    bool isKeyboardCaptured();
    bool isVisible();
    bool forceCursor() const;
    void setGUIState(bool on);

    void darkMode();

    void registerMouseCaptureStateChangedHandler(std::function<void(bool)> handler);
    void registerKeyboardCaptureStateChangedHandler(std::function<void(bool)> handler);

    template <typename... T>
    void addStatistic(const std::string& blockName, std::tuple<T...> stat) {
        std::apply([&](auto&&... args){ ((ss << std::move(args)), ...); }, std::move(stat));
        statistics[blockName].push_back(ss.str());
        ss.clear();
        ss.str("");
    }

private:
    void createDescriptorPool();
    void createRenderPass();
    void createFramebuffers();

private:
    GUIManager() = default;

private:
    vk::DescriptorPool descriptorPool;
    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> frameBuffers;
    bool isInitialized = false;
    bool showUI = true, rdp = false;
    ImGuiIO* pIo = nullptr;
    ImGuiContext* pMainContext = nullptr;
    std::vector<std::function<void(bool)>> mouseCaptureStateChangedHandlers;
    std::vector<std::function<void(bool)>> keyboardCaptureStateChangedHandlers;

    std::unordered_map<std::string, std::vector<std::string>> statistics;
    std::stringstream ss;

private:
    WindowManager* wm = nullptr;
};

inline auto& theGUIManager = GUIManager::Instance();

template <typename F>
void gui(F&& f) {
    if (theGUIManager.isVisible()) {
        f();
    }
}

#endif//VULKAN_INTRO_GUI_MANAGER_H
