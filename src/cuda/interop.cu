#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "interop.cuh"
#include "stdio.h"
#include <memory>
#include <cuda/std/complex>
#include "../glfwim/input_manager.h"
#include "../camera/mandelbrotCamera.h"
#include "../camera/perspective_camera.h"

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

uint32_t imageWidth, imageHeight;
cudaExternalMemory_t cudaExtMemImageBuffer; // memory handler to the imported memory allocation
cudaMipmappedArray_t cudaMipmappedImageArray; // the image interpreted as a mipmapped array
cudaSurfaceObject_t surfaceObject; // surface object to the first mip level of the array. Allows write

void freeExportedVulkanImage()
{
    checkCudaError(cudaDestroySurfaceObject(surfaceObject));
    checkCudaError(cudaFreeMipmappedArray(cudaMipmappedImageArray));
    checkCudaError(cudaDestroyExternalMemory(cudaExtMemImageBuffer));
}

void exportVulkanImageToCuda_R8G8B8A8Unorm(void* mem, VkDeviceSize size, VkDeviceSize offset, uint32_t width, uint32_t height)
{
    imageWidth = width;
    imageHeight = height;

    // import memory into cuda through native handle (win32)
    cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
    memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
    cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32; // after win8
    cudaExtMemHandleDesc.handle.win32.handle = mem; // allocation handle
    cudaExtMemHandleDesc.size = size; // allocation size
   
    checkCudaError(cudaImportExternalMemory(&cudaExtMemImageBuffer, &cudaExtMemHandleDesc));

    // extract mipmapped array from memory
    cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;
    memset(&externalMemoryMipmappedArrayDesc, 0, sizeof(externalMemoryMipmappedArrayDesc));

    // we want ot interpret the raw memory as an image so we need to specify its format and layout
    cudaExtent extent = make_cudaExtent(width, height, 0);
    cudaChannelFormatDesc formatDesc; // 4 channel, 8 bit per channel, unsigned
    formatDesc.x = 8;
    formatDesc.y = 8;
    formatDesc.z = 8;
    formatDesc.w = 8;
    formatDesc.f = cudaChannelFormatKindUnsigned;

    externalMemoryMipmappedArrayDesc.offset = offset; // the image starts here
    externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
    externalMemoryMipmappedArrayDesc.extent = extent;
    externalMemoryMipmappedArrayDesc.flags = 0;
    externalMemoryMipmappedArrayDesc.numLevels = 1; // no mipmapping
    checkCudaError(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArray, cudaExtMemImageBuffer, &externalMemoryMipmappedArrayDesc));

    // extract first level
    cudaArray_t cudaMipLevelArray;
    checkCudaError(cudaGetMipmappedArrayLevel(&cudaMipLevelArray, cudaMipmappedImageArray, 0));

    // create surface object for writing
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaMipLevelArray;
    
    checkCudaError(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));
}

cudaExternalSemaphore_t cudaWaitsForVulkanSemaphore, vulkanWaitsForCudaSemaphore;

void freeExportedSemaphores()
{
    checkCudaError(cudaDestroyExternalSemaphore(cudaWaitsForVulkanSemaphore));
    checkCudaError(cudaDestroyExternalSemaphore(vulkanWaitsForCudaSemaphore));
}

void exportSemaphoresToCuda(void* cudaWaitsForVulkanSemaphoreHandle, void* vulkanWaitsForCudaSemaphoreHandle) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.flags = 0;
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;

    externalSemaphoreHandleDesc.handle.win32.handle = cudaWaitsForVulkanSemaphoreHandle;
    checkCudaError(cudaImportExternalSemaphore(&cudaWaitsForVulkanSemaphore, &externalSemaphoreHandleDesc));

    externalSemaphoreHandleDesc.handle.win32.handle = vulkanWaitsForCudaSemaphoreHandle;
    checkCudaError(cudaImportExternalSemaphore(&vulkanWaitsForCudaSemaphore, &externalSemaphoreHandleDesc));
}

// compresses 4 32bit floats into a 32bit uint
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
        ((unsigned int)(rgba.z * 255.0f) << 16) |
        ((unsigned int)(rgba.y * 255.0f) << 8) |
        ((unsigned int)(rgba.x * 255.0f));
}

__device__ float map(float x, float fromMin, float fromMax, float toMin, float toMax) {
    return toMin + (toMax - toMin) * (x - fromMin) / (fromMax - fromMin);
}


/*
    While and Nylander's formula for the "nth power" of the vector v = (x, y, z)
*/
__device__ float3 vectorPower(float3 v, unsigned int n)
{
    float r = norm3df(v.x, v.y, v.z);
    float psi = atan2f(v.y, v.x);
    float theta = acosf(v.z / r);
    float nf = (float)n;
    float powRN = powf(r, nf);
    return float3(powRN * sinf(nf * theta) * cosf(nf * psi), powRN * sinf(nf * theta) * sinf(nf * psi), powRN * cosf(nf * theta));
}


__device__ float4 mandelbulb(float3 c, unsigned int n, unsigned int iterationLimit, float pseudoInfinity)
{
    float4 sampleOut = float4{ 0.9f, 0.99f, 0.92f, 0.5f };
    
    float3 v = float3(0.0f, 0.0f, 0.0f);

    for (unsigned int i = 0; i < iterationLimit; i++) {
        float3 powV = vectorPower(v, n);
        v = float3(powV.x + c.x, powV.y + c.y, powV.z + c.z);
        if (norm3df(v.x, v.y, v.z) > pseudoInfinity) {
            sampleOut = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
            break;
        }
    }
    return sampleOut;
}


__global__ void renderToSurface(cudaSurfaceObject_t dstSurface, size_t width, size_t height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float wx = map(float(x), 0.0f, (float)width, 0.0f, 1.0f);
    float wy = map(float(y), 0.0f, (float)height, 0.0f, 1.0f);

    float4 dataOut = float4{ wx, wy, wx, 1.0f };

    surf2Dwrite(rgbaFloatToInt(dataOut), dstSurface, x * 4, y); // expects byte coordinates. 1 pixel = 4 byte
}

// Black and White plot
__global__ void renderMandelbrotToSurface(cudaSurfaceObject_t dstSurface, size_t width, size_t height, float pos_x, float pos_y, float zoom)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float wx = map(float(x), 0.0f, (float)width, (-2.0f + pos_x) / zoom, (2.0f + pos_x) / zoom);
    float wy = map(float(y), 0.0f, (float)height, (-2.0f + pos_y) / zoom, (2.0f + pos_y) / zoom);
    
    cuda::std::complex<float> c = cuda::std::complex<float>(wx, wy);
    cuda::std::complex<float> z = 0.0f;
    int n = 100;
    double infinity = 200;
    float4 dataOut = float4{ wx, wy, wx, 1.0f };    // Default (For debug)
    dataOut = float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    for (unsigned int escapeTime = 0; escapeTime < n; escapeTime++) {
        z = z*z + c;
        if (abs(z) > infinity) {
            float fractional = logf(logf(abs(z)) / logf(infinity)) / logf(2.0);
            glm::vec3 color = 0.5f + 0.5f * cos(3.0f + (escapeTime - fractional) * 0.15f + glm::vec3(0.0f, 0.6f, 1.0));    // Coloring
            dataOut = float4{ color.x, color.y, color.z, 1.0f };
            break;
        }
    }
    surf2Dwrite(rgbaFloatToInt(dataOut), dstSurface, x * 4, y); // expects byte coordinates. 1 pixel = 4 byte
}


void renderCuda()
{
    uint32_t nthreads = 32;
    dim3 dimBlock{ nthreads, nthreads };
    dim3 dimGrid{ imageWidth / nthreads + 1, imageHeight / nthreads + 1 };
    renderMandelbrotToSurface<<<dimGrid, dimBlock>>>(surfaceObject, imageWidth, imageHeight, MandelbrotX, MandelbrotY, MandelbrotZoom);
    checkCudaError(cudaGetLastError());
    //checkCudaError(cudaDeviceSynchronize()); // not optimal! should be synced with vulkan using semaphores
}