#ifndef VULKAN_INTRO_APPLICATION_H
#define VULKAN_INTRO_APPLICATION_H

#include "camera/perspective_camera.h"
#include "camera/mandelbrotCamera.h"
#include "rendering/renderer.h"
#include "window_manager.h"
#include "scene/game_scene.h"

class GameObject;
class PointLight;
class AmbientLight;

class Application {
public:
    static Application& instance() {
        static Application app;
        return app;
    }

    void initialize();
    void destroy();
    void start();

private:
    Application() = default;
    void handleInput();
    void update(double dt);
    void updateGui();
    bool draw();

    void initScene();

    void initializeSystems();
    void destroySystems();
    bool initializeSwapChain();
    void destroySwapChain();
    bool recreateSwapChain();

private:
    int maxFPS = 60;
    float logicalFramesPerSec = 60.0;
    bool swapChainValid = false;

public:
    Renderer renderer;
    PerspectiveCamera camera;
    MandelbrotCamera mCamera;
    GameScene gameScene;
};

inline auto& theApplication = Application::instance();

#endif//VULKAN_INTRO_APPLICATION_H
