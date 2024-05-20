#ifndef RENDER_ENGINE_PCH_H
#define RENDER_ENGINE_PCH_H

#ifdef __cplusplus

#define NOMINMAX
#define _USE_MATH_DEFINES

#include <filesystem>
#include <thread>
#include <future>
#include <atomic>
#include <string>
#include <string_view>
#include <cassert>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <set>
#include <fstream>
#include <numeric>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <functional>
#include <optional>
#include <unordered_set>
#include <random>
#include <list>
#include <limits>
#include <queue>
#include <type_traits>
#include <variant>
#include <ranges>
#include <numbers>

#define LOG_LEVELS(CHAR, NAME_STANDARD, NAME_UPPER, VALUE) \
    CHAR('V') NAME_STANDARD(Vulkan) NAME_UPPER(VULKAN) VALUE(1) \
    CHAR('F') NAME_STANDARD(Fatal) NAME_UPPER(FATAL) VALUE(4) \
    CHAR('E') NAME_STANDARD(Error) NAME_UPPER(ERROR) VALUE(8) \
    CHAR('W') NAME_STANDARD(Warning) NAME_UPPER(WARN) VALUE(16) \
    CHAR('I') NAME_STANDARD(Info) NAME_UPPER(INFO) VALUE(32) \
    CHAR('D') NAME_STANDARD(Debug) NAME_UPPER(DEBUG) VALUE(64)

#define EZPZLOGGER_USE_LOCK
#include <EZPZLogger/logger.hpp>
extern ezpz::Logger::LoggerWrapper vulkanLogger;
extern ezpz::Logger::LoggerWrapper logger;


using namespace std::literals::chrono_literals;
namespace rs = std::views;

#pragma warning(push, 0)  

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
//#define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
//       printf((format), __VA_ARGS__); \
//       printf("\n"); \
//   } while(false)
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/vec_swizzle.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <json/json.hpp>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui/imgui.h>
#include <imgui/imgui_stdlib.h>
#include <imgui/ImGuizmo.h>

#pragma warning(pop)

#include <glfwim/input_manager.h>

#endif

#endif // RENDER_ENGINE_PCH_H
