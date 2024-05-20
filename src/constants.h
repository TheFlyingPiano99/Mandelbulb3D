#ifndef VULKAN_INTRO_CONSTANTS_H
#define VULKAN_INTRO_CONSTANTS_H

#define MAX_FRAMES_IN_FLIGHT 2
#define MAX_OBJECT_NUM 200

inline std::filesystem::path cDirShader = "shaders";
inline std::filesystem::path cDirShaderSources = "../src/shaders";
inline std::filesystem::path cDirCache = "cache";
inline std::filesystem::path cDirModels = "models";
inline bool cLogAssetLoading = false;
inline bool cDebugValidationLayers = true;
inline bool cDebugObjectNames = true;

#endif//VULKAN_INTRO_CONSTANTS_H
