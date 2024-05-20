#ifndef VULKAN_INTRO_GAME_SCENE_H
#define VULKAN_INTRO_GAME_SCENE_H

#include "scene/lights.h"
#include "game_object.h"

class GameScene {
public:
    std::vector<GameObject> gameObjects;
    std::vector<PointLight> pointLights;
    AmbientLight ambientLight;

    void updateGui();

private:
    size_t selectedObjectIndex = 0, selectedPointLightIndex = 0;
};

#endif//VULKAN_INTRO_GAME_SCENE_H
