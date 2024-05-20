#ifndef VULKAN_INTRO_RENDER_CONTEXT_H
#define VULKAN_INTRO_RENDER_CONTEXT_H

class GameScene;
class PerspectiveCamera;

struct RenderContext
{
    GameScene* pGameScene;
    PerspectiveCamera* cam;
    vk::CommandBuffer cmd;
    size_t frameID, imageID;
};

#endif//VULKAN_INTRO_RENDER_CONTEXT_H
