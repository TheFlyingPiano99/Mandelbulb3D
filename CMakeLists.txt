cmake_minimum_required(VERSION 3.16)
project(vulkanCudaInterop LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 23)

add_compile_definitions(USE_EZPZLOGGER)

list(APPEND THIRDPARTY_FILES
    thirdparty/imgui/imgui.cpp
    thirdparty/imgui/imgui_draw.cpp
    thirdparty/imgui/imgui_stdlib.cpp
    thirdparty/imgui/imgui_widgets.cpp
    thirdparty/imgui/imgui_tables.cpp
    thirdparty/imgui/imgui_impl_glfw.cpp
    thirdparty/imgui/imgui_impl_vulkan.cpp
    thirdparty/imgui/imgui_demo.cpp
    thirdparty/imgui/ImGuizmo.cpp
    thirdparty/glfwim/input_manager_impl.cpp
    thirdparty/glfwim/input_manager.cpp)

list(APPEND RESOURCE_FILES
    src/resources/gpu_resources.cpp
    src/resources/resource_manager.cpp)

list(APPEND ASSET_MANAGMENT_FILES
    src/asset_managment/asset_manager.cpp
    src/asset_managment/asset_loader.cpp

    src/asset_managment/texture.cpp
    src/asset_managment/mesh.cpp
    src/asset_managment/material.cpp
    src/asset_managment/coloured_mesh.cpp
    src/asset_managment/composite_mesh.cpp

    src/asset_managment/material_traits.cpp

    src/asset_managment/asset_loader_gltf.cpp
    src/asset_managment/asset_loader_obj.cpp
    src/asset_managment/asset_loader_texture.cpp)

list(APPEND RENDERER_FILES
    src/rendering/renderer.cpp
    src/shader_manager.cpp
    src/rendering/common_rendering.cpp)

list(APPEND CUDA_FILES
    src/cuda/interop.cuh
    src/cuda/interop.cu)

list(APPEND MAIN_FILES
    src/camera/perspective_camera.cpp
    src/scene/game_scene.cpp
    src/main.cpp
    src/application.cpp
    src/vulkan_engine.cpp
    src/window_manager.cpp
    src/vulkan_layer.cpp
    src/gui/gui_manager.cpp
    src/utility/utility.cpp
    src/gui/imgui_ext.cpp
    src/utility/mmath.cpp)

add_executable(${PROJECT_NAME} 
    ${MAIN_FILES} ${RENDERER_FILES} ${CUDA_FILES}
    ${ASSET_MANAGMENT_FILES} ${RESOURCE_FILES} ${THIRDPARTY_FILES} "src/camera/mandelbrotCamera.h" "src/camera/mandelbrotCamera.cpp" "src/mandelbulb/mandelbulb_animator.h" "src/mandelbulb/mandelbulb_animator.cpp")
set_source_files_properties(${THIRDPARTY_FILES} PROPERTIES COMPILE_OPTIONS -w)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "native")
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include") # intellisense
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # dynamic parallelism
set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_OPTIONS -lineinfo) # nsight source map

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
#find_package(glfw3 3.3 REQUIRED) # uncomment on linux

target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.h)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE thirdparty thirdparty/imgui thirdparty/glm/include ${Vulkan_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE thirdparty/glfw/include) # comment on linux

list(APPEND LIBRARIES
    ${Vulkan_LIBRARIES}
    glfw3 # comment on linux
    #glfw # uncomment on linux
    Threads::Threads)

if (MSVC)
    #target_compile_options(${PROJECT_NAME} PRIVATE /Zi)
    target_link_directories(${PROJECT_NAME} PRIVATE thirdparty/glfw/lib-vc2022)
    #target_link_options(${PROJECT_NAME} PRIVATE /INCREMENTAL /SAFESEH:NO)
else()
    target_link_directories(${PROJECT_NAME} PRIVATE thirdparty/glfw/lib-mingw-w64) # commment on linux
endif()

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
