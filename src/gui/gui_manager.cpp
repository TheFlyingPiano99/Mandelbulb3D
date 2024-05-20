#include "gui_manager.h"
#include "vulkan_layer.h"
#include "constants.h"
#include "window_manager.h"
#include "rendering/render_context.h"
#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>
#include <GLFW/glfw3.h>
#include "gui/imgui_ext.h"

void GUIManager::initialize(WindowManager* pWindowManager)
{
    wm = pWindowManager;

    IMGUI_CHECKVERSION();
    pMainContext = ImGui::CreateContext();
    pIo = &ImGui::GetIO(); (void)*pIo;
    pIo->ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    pIo->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    darkMode();
    ImGuiStyle& style = ImGui::GetStyle();
    style.CircleTessellationMaxError = 0.1f;
    if (pIo->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
    }
    ImGui_ImplGlfw_InitForVulkan(wm->window, false);

    theInputManager.setDefaultHandlerPriority(0); 

    theInputManager.registerWindowFocusHandler([w = wm->window](bool focused) {
        ImGui_ImplGlfw_WindowFocusCallback(w, (int)focused);
    });

    theInputManager.registerCursorPositionHandler([w = wm->window](double x, double y) {
        ImGui_ImplGlfw_CursorPosCallback(w, x, y);
    });

    theInputManager.registerCursorMovementHandler([w = wm->window](glfwim::CursorMovement mov) {
        ImGui_ImplGlfw_CursorEnterCallback(w, (int)mov);
    });

    theInputManager.registerMouseButtonHandler([w = wm->window](auto button, auto mod, auto action) {
        ImGui_ImplGlfw_MouseButtonCallback(w, (int)button, (int)action, (int)mod);
    });

    theInputManager.registerMouseScrollHandler([w = wm->window](double x, double y) {
        ImGui_ImplGlfw_ScrollCallback(w, x, y);
    });

    theInputManager.registerKeyHandlerWithKey([w = wm->window](int key, auto mod, auto action) {
        ImGui_ImplGlfw_KeyCallback(w, key, glfwGetKeyScancode(key), (int)action, (int)mod);
    });

    theInputManager.registerTextCallback([w = wm->window](auto keypoint) {
         ImGui_ImplGlfw_CharCallback(w, keypoint);
    });

    theInputManager.registerMonitorStateChangedHandler([](auto monitor, auto event) {
        ImGui_ImplGlfw_MonitorCallback(monitor, event);
    });

    theInputManager.setDefaultHandlerPriority(1);

    createDescriptorPool();
}

void GUIManager::initializeSwapChainDependentComponents()
{
    auto& vl = theVulkanLayer;
    // Hack: imgui does not like multiple calls to ImGui_ImplVulkan_Init/Shutdown but we need it to be in swapchain dependent initialization
    static bool first = true;
    if (first) {
        createRenderPass();

        ImGui_ImplVulkan_InitInfo info {};

        info.Instance = (VkInstance)vl.instance;
        info.PhysicalDevice = (VkPhysicalDevice)vl.physicalDevice;
        info.Device = (VkDevice)vl.device;
        info.graphicsSubmitFunction = [](const VkSubmitInfo& info, VkFence fence) { return (VkResult)theVulkanLayer.safeSubmitGraphicsCommand(info, false, fence); };
        info.presentSubmitFunction = [](const VkPresentInfoKHR& info) {
            try { 
                return (VkResult)theVulkanLayer.safeSubmitPresentCommand(info);
            } catch (...) {
                return VkResult::VK_ERROR_OUT_OF_DATE_KHR; // it throws exception in this case so we got to catch it
            }
        };
        info.waitIdleFunction = []()
        {
            try {
                theVulkanLayer.safeWaitIdle();
                return VkResult::VK_SUCCESS;
            } catch (...) {
                return VkResult::VK_ERROR_OUT_OF_DATE_KHR; // it throws exception in this case so we got to catch it
            }
        };
        //info.Queue = (VkQueue)vl.graphicsQueue;
        info.DescriptorPool = (VkDescriptorPool)descriptorPool;
        info.MinImageCount = (uint32_t)wm->swapChainImageViews.size();
        info.ImageCount = (uint32_t)wm->swapChainImageViews.size();
        info.PipelineCache = (VkPipelineCache)vl.pipelineCache;
        info.MSAASamples = VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT;
        info.QueueFamily = vl.graphicsFamilyIndex;
        info.CheckVkResultFn = [](VkResult err) {
            if (err != VK_SUCCESS) {
                vulkanLogger.LogFatal("Error in imgui. DEBUG ME");
            }
        };

        if (!ImGui_ImplVulkan_Init(&info, (VkRenderPass)renderPass)){
            throw std::runtime_error("Failed to initialize ImGui!");
        }

        ImGuiIO* io = &ImGui::GetIO();
        io->Fonts->Build();

        auto cmd = vl.beginSingleTimeCommands();
        if (!ImGui_ImplVulkan_CreateFontsTexture((VkCommandBuffer)cmd)){
            vl.endSingleTimeCommands(cmd);
            throw std::runtime_error("Failed to initialize ImGui (fonts)!");
        }
        vl.endSingleTimeCommands(cmd);
        ImGui_ImplVulkan_DestroyFontUploadObjects();

        first = false;
    }

    createFramebuffers();
    isInitialized = true;
}

void GUIManager::destroy()
{
    auto& vl = theVulkanLayer;

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(pMainContext);

    vl.device.destroy(descriptorPool);
    vl.device.destroy(renderPass);
}

void GUIManager::destroySwapChainDependentComponents()
{
    auto& vl = theVulkanLayer;
    isInitialized = false;
    for (auto fb : frameBuffers){
        vl.device.destroy(fb);
    }
    frameBuffers.clear();
}

void GUIManager::updateGui()
{
    if (ImGui::Begin("Settings")) {
        ImGui::Checkbox("RDP", &rdp);
    }
    ImGui::End();

    if (ImGui::Begin("Statistics")) {
        for (auto& it : statistics) {
            if (it.second.empty()) continue;
            ImGui::Text("%s", it.first.c_str());
            ImGui::Separator();
            for (auto& t : it.second) {
                ImGui::Text("%s", t.c_str());
            }
            ImGui::Separator();
            it.second.clear();
        }
    }
    ImGui::End();
}

void GUIManager::startFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    bool wasCapturedMouse = isMouseCaptured();
    bool wasCapturedKeyboard = isKeyboardCaptured();

    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
    ImGuizmo::ResetGlobalID();

    if (wasCapturedMouse != isMouseCaptured()) {
        theInputManager.setCurrentMouseHandlerPriority(wasCapturedMouse);
        for (auto& h : mouseCaptureStateChangedHandlers){
            h(isMouseCaptured());
        }
    }

    if (wasCapturedKeyboard != isKeyboardCaptured()) {
        theInputManager.setCurrentKeyboardHandlerPriority(wasCapturedKeyboard);
        for (auto& h : keyboardCaptureStateChangedHandlers){
            h(isKeyboardCaptured());
        }
    }
}

void GUIManager::finishFrame(bool skipRender)
{
    ImGui::EndFrame();

    if (pIo->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        wm->backupCurrentContext();
        ImGui::UpdatePlatformWindows();
        if (!skipRender) ImGui::RenderPlatformWindowsDefault();
        wm->restoreCurrentContext();
    }
}

void GUIManager::render(const RenderContext& ctx)
{
    ImGui::Render();

    vk::RenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = frameBuffers[ctx.imageID];
    renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
    renderPassInfo.renderArea.extent = wm->swapChainExtent;

    std::array<vk::ClearValue, 1> clearValues = {};
    clearValues[0].color = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.f}};

    renderPassInfo.clearValueCount = clearValues.size();
    renderPassInfo.pClearValues = clearValues.data();

    ctx.cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    if (showUI)
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), (VkCommandBuffer)ctx.cmd);

    ctx.cmd.endRenderPass();
}

bool GUIManager::isMouseCaptured()
{
    return isInitialized && pIo->WantCaptureMouse;
}

bool GUIManager::isKeyboardCaptured()
{
    return isInitialized && pIo->WantCaptureKeyboard;
}

void GUIManager::registerMouseCaptureStateChangedHandler(std::function<void(bool)> handler)
{
    mouseCaptureStateChangedHandlers.push_back(std::move(handler));
}

void GUIManager::registerKeyboardCaptureStateChangedHandler(std::function<void(bool)> handler)
{
    keyboardCaptureStateChangedHandlers.push_back(std::move(handler));
}

void GUIManager::createDescriptorPool()
{
    auto& vl = theVulkanLayer;
    std::array<vk::DescriptorPoolSize, 1> poolSizes = {};
    poolSizes[0].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[0].descriptorCount = 1;

    vk::DescriptorPoolCreateInfo info = {};
    info.poolSizeCount = poolSizes.size();
    info.pPoolSizes = poolSizes.data();
    info.maxSets = MAX_FRAMES_IN_FLIGHT;

    descriptorPool = vl.device.createDescriptorPool(info);
}

void GUIManager::createRenderPass()
{
    auto& vl = theVulkanLayer;

    vk::AttachmentDescription attachment = {};

    attachment.format = vl.swapChainMutableFormatAvailable ? vk::Format::eB8G8R8A8Unorm :  wm->swapChainImageFormat;
    attachment.samples = vk::SampleCountFlagBits::e1;
    attachment.loadOp = vk::AttachmentLoadOp::eLoad;
    attachment.storeOp = vk::AttachmentStoreOp::eStore;
    attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachment.initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
    attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_attachment = {};
    color_attachment.attachment = 0;
    color_attachment.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass = {};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment;

    vk::SubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = {};
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo info = {};
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;

    renderPass = vl.name(vl.device.createRenderPass(info), "ImguiRenderPass");
}

void GUIManager::createFramebuffers()
{
    auto& vl = theVulkanLayer;

    frameBuffers.reserve(wm->swapChainImageViewsUnorm.size());
    for (auto& swapChainImageView : wm->swapChainImageViewsUnorm) {
        std::array<vk::ImageView, 1> attachments = {
                swapChainImageView
        };

        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = wm->swapChainExtent.width;
        framebufferInfo.height = wm->swapChainExtent.height;
        framebufferInfo.layers = 1;

        frameBuffers.push_back(vl.device.createFramebuffer(framebufferInfo));
    }
}

bool GUIManager::isVisible()
{
    return showUI;
}

void GUIManager::setGUIState(bool on)
{
    showUI = on;
}

bool GUIManager::forceCursor() const
{
    return rdp;
}

void GUIManager::darkMode()
{
    ImVec4* colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
    colors[ImGuiCol_Border] = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.071f, 0.064f, 0.064f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.036f, 0.031f, 0.076f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.071f, 0.064f, 0.064f, 1.00f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_Button] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
    colors[ImGuiCol_Separator] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_DockingEmptyBg] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(8.00f, 8.00f);
    style.FramePadding = ImVec2(5.00f, 2.00f);
    style.CellPadding = ImVec2(6.00f, 6.00f);
    style.ItemSpacing = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding = ImVec2(0.00f, 0.00f);
    style.IndentSpacing = 25;
    style.ScrollbarSize = 15;
    style.GrabMinSize = 10;
    style.WindowBorderSize = 1;
    style.ChildBorderSize = 1;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = 1;
    style.TabBorderSize = 1;
    style.WindowRounding = 7;
    style.ChildRounding = 4;
    style.FrameRounding = 7;
    style.PopupRounding = 4;
    style.ScrollbarRounding = 9;
    style.GrabRounding = 3;
    style.LogSliderDeadzone = 4;
    style.TabRounding = 4;
}
