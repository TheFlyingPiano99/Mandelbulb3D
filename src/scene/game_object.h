#ifndef VULKAN_INTRO_GAME_OBJECT_H
#define VULKAN_INTRO_GAME_OBJECT_H

#include "transform.hpp"
#include "asset_managment/composite_mesh.h"

struct GameObject {
	Transform transform;
	CompositeMeshHandler mesh;
	std::string name;
};

#endif//VULKAN_INTRO_GAME_OBJECT_H