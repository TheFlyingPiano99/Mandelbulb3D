#ifndef CUDA_INTEROP_CUH
#define CUDA_INTEROP_CUH
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

// Exports a vulkan memory allocation into CUDA. 
// mem - native win32 handle to the memory allocation
// size - entire size of the allocation
// offset - offset to the start of the image to be used by CUDA
// width, height - image dimensions
// the expected format is hardcoded in the function body (and in the function name)
void exportVulkanImageToCuda_R8G8B8A8Unorm(void* mem, VkDeviceSize size, VkDeviceSize offset, uint32_t width, uint32_t height);
// frees the exported resources
// the resources are stored in global variables for simplicity so resources must be freed before a new export
void freeExportedVulkanImage();
void renderCuda();

void freeExportedSemaphores();
void exportSemaphoresToCuda(void* cudaWaitsForVulkanSemaphoreHandle, void* vulkanWaitsForCudaSemaphoreHandle);

#endif //CUDA_INTEROP_CUH
