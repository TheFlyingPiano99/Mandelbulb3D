#ifndef VULKAN_INTRO_CONSTANTS_H
#define VULKAN_INTRO_CONSTANTS_H

#define MAX_FRAMES_IN_FLIGHT 2
#define MAX_OBJECT_NUM 200

inline std::filesystem::path cDirShader = std::filesystem::path(WORK_DIR).append("shaders");
inline std::filesystem::path cDirShaderSources = std::filesystem::path(WORK_DIR).append("../src/shaders");
inline std::filesystem::path cDirCache = std::filesystem::path(WORK_DIR).append("cache");
inline std::filesystem::path cDirModels = std::filesystem::path(WORK_DIR).append("models");
inline bool cLogAssetLoading = false;
inline bool cDebugValidationLayers = true;
inline bool cDebugObjectNames = true;

#endif//VULKAN_INTRO_CONSTANTS_H
