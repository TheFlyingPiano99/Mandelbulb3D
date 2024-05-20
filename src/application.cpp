#include "application.h"
#include "vulkan_layer.h"
#include "vulkan_engine.h"
#include "resources/resource_manager.hpp"
#include <GLFW/glfw3.h>
#include "shader_manager.h"
#include "utility/utility.hpp"
#include "asset_managment/asset_loader.h"
#include "asset_managment/asset_manager.h"
#include "scene/game_scene.h"
#include "gui/gui_manager.h"

using namespace glfwim;
namespace fs = std::filesystem;

void Application::initialize()
{
    if (!fs::is_directory("cache")) {
        try {
            fs::create_directory("cache");
        } catch (std::exception& e) {
            logger.LogError("Could not create cache directory: ", e.what());
        }
    }

    initializeSystems();
    initializeSwapChain();
    initScene();

    thePerspectiveCamera.registerHandlers();

    theInputManager.registerUtf8KeyHandler("\n", Modifier::Alt, Action::Press, [&](){ theWindowManager.toggleFullscreen(); });
    theInputManager.registerUtf8KeyHandler("u", Modifier::None, Action::Press, [&](){ theGUIManager.setGUIState(!theGUIManager.isVisible()); });
}

void Application::destroy()
{
    destroySwapChain();
    destroySystems();
}

void Application::start()
{
    auto renderThread = std::thread{ [this]() {
        double lastUpdate = 0.0;
        double timeSLU = 0.0;
        double lastDraw = 0.0;
        glfwSetTime(timeSLU);
        double timePerFrame = 1.0 / logicalFramesPerSec;

        try {
            // game loop
            while (!theWindowManager.shouldClose()) {
                double actTime = glfwGetTime();
                timeSLU += actTime - lastUpdate;
                lastUpdate = actTime;

                // input handling and game state update
                while (timeSLU > timePerFrame) {
                    timeSLU -= timePerFrame;
                    theInputManager.handleEvents();
                    update(timePerFrame);
                }

                bool canDraw = true;
                if (!theWindowManager.ready()) {
                    canDraw = false;
                    if (theWindowManager.stable()) {
                        canDraw = recreateSwapChain();
                    }
                }
                
                // drawing
                if (actTime - lastDraw > 1.0 / (double)maxFPS) {
                    lastDraw = glfwGetTime();
                    theGUIManager.startFrame();

                    ImGui::ShowDemoWindow();

                    gui([&]() {updateGui(); });
                    bool swapChainIntact = true;

                    if (canDraw) swapChainIntact = draw();
                    else ImGui::Render();

                    theGUIManager.finishFrame(!swapChainIntact);
                } else {
                    std::this_thread::sleep_for(std::chrono::seconds{ (int)(1.0 / (double)maxFPS - (actTime - lastDraw)) } - std::chrono::milliseconds{ 10 });
                }
            }
        } catch(std::exception& e) {
            logger.LogError("Unhandled exception caught in render thread:\n", e.what(), "\nTerminating...");
        } catch (...) {
            logger.LogError("Unkown unhandled exception caught in render thread. Terminating...");
        }

        update(timePerFrame);

        theVulkanEngine.waitIdle();

        theWindowManager.close();
    }};

    while (!theWindowManager.shouldClose()) {
        handleInput();
    }

    renderThread.join();
}

void Application::handleInput()
{
    theInputManager.pollEvents();
    theInputManager.runTasks();
}

void Application::update(double dt)
{
    if (theWindowManager.isVisible()) {
        renderer.update(dt);
        thePerspectiveCamera.update(dt);
        theMandelbulbAnimator.update(dt);
    }
}

void Application::updateGui()
{
    if (ImGui::Begin("Settings")) {
        ImGui::DragInt("Max FPS", &maxFPS, 1.0, 16, 512);
        ImGui::DragFloat("Logical FPS", &logicalFramesPerSec, 0.5, 16, 512);
    }
    ImGui::End();
    gameScene.updateGui();
    theGUIManager.updateGui();
    theVulkanEngine.updateGui();
    thePerspectiveCamera.updateGui();
    theMandelbulbAnimator.updateGui();
}

bool Application::draw()
{
    auto result = theVulkanEngine.drawFrame(&renderer, &thePerspectiveCamera, &gameScene);
    switch (result){
        case VulkanEngine::Result::Success: return true;
        case VulkanEngine::Result::SwapChainOutOfDate: theWindowManager.invalidate(); return false;
    }
    assert(false);
    return true;
}

void Application::initScene()
{
    if (fs::exists(cDirModels)) {
        theAssetLoader.addRootAssetLocationDirectory(cDirModels.string());
        theAssetLoader.collectFiles();
    } else {
        logger.LogError("Models directory does not exist: ", cDirModels);
    }

    theAssetLoader.validateAssets({ "BrainStem.glb" }, theCompatibilityDescriptor);
    theAssetLoader.validateAssets({ "sponza/Sponza.glb" }, theCompatibilityDescriptor);

    thePerspectiveCamera.moveTo(glm::vec3{ 5, 5, 5 });
    thePerspectiveCamera.lookAt(glm::vec3(0, 0, 0));

    gameScene.ambientLight.color = glm::vec3(1.0);
    gameScene.ambientLight.power = 0.1;
}

void Application::initializeSystems()
{
    theVulkanLayer.initializeVulkan();
    theWindowManager.initializeWindow();
    theVulkanLayer.initializeDevices(theWindowManager.surface);
    theWindowManager.prepareSwapChain();
    theInputManager.initialize(theWindowManager.window);
    theInputManager.setDefaultHandlerPriority(1);
    theInputManager.setCurrentKeyboardHandlerPriority(1);
    theInputManager.setCurrentMouseHandlerPriority(1);
    theWindowManager.registerCallbacks();

    theShaderManager.initialize();
    theResourceManager.initialize();
    theAssetManager.initialize();
    theAssetLoader.initialize();
    renderer.initialize();
    theVulkanEngine.initialize();
    theGUIManager.initialize(&theWindowManager);
}

void Application::destroySystems()
{
    theGUIManager.destroy();
    renderer.destroy();
    theVulkanEngine.destroy();
    theAssetLoader.destroy();
    theAssetManager.destroy();
    theResourceManager.destroy();

    theVulkanLayer.destroyDevices();
    theWindowManager.destroyWindow();
    theVulkanLayer.destroyVulkan();
}

bool Application::initializeSwapChain()
{
    bool ret = theWindowManager.initializeSwapChain();
    if (!ret) return false;
    theGUIManager.initializeSwapChainDependentComponents();
    renderer.initializeSwapChainDependentComponents();
    return swapChainValid = true;
}

void Application::destroySwapChain()
{
    renderer.destroySwapChainDependentComponents();
    theGUIManager.destroySwapChainDependentComponents();
    theWindowManager.destroySwapChain();
    swapChainValid = false;
}

bool Application::recreateSwapChain()
{
    theVulkanEngine.waitIdle();
    if (swapChainValid) {
        destroySwapChain();
    }
    return initializeSwapChain();
}
