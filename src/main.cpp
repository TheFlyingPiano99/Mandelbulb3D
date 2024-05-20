#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#define TINYGLTF_USE_CPP14
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_IMPLEMENTATION
#include <stbi/stb_image.h>
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#endif
#include <tinygltf/tiny_gltf.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnullability-completeness"
#endif
#include <vma/vk_mem_alloc.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#undef ERROR
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#define LOGGER_IMPLEMENTATION
#include <EZPZLogger/logger.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
ezpz::Logger::LoggerWrapper vulkanLogger = ezpz::Logger::Instance().CreateWrapper(ezpz::LogLevel::VULKAN);
ezpz::Logger::LoggerWrapper logger = ezpz::Logger::Instance().CreateWrapper(ezpz::LogLevel{});
#undef _CRT_SECURE_NO_WARNINGS

#include "application.h"

int main() {
    ezpz::theLogger.Init("logger_settings.ini");

    Application& app = theApplication;
    app.initialize();

    auto ret = EXIT_SUCCESS;

    try {
        app.start();
    } catch (const std::exception& e) {
        logger.LogError("Unhandled exception caught:\n", e.what(), "\nTerminating...");
        ret = EXIT_FAILURE;
    } catch (...) {
        logger.LogError("Unkown unhandled exception caught. Terminating...");
        ret = EXIT_FAILURE;
    }

    app.destroy();
    
    return ret;
}
