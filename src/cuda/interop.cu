#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "interop.cuh"
#include "stdio.h"
#include <memory>
#include <cuda/std/complex>
#include "../glfwim/input_manager.h"
#include "../camera/perspective_camera.h"
#include "../mandelbulb/mandelbulb_animator.h"
#include "glm/gtc/matrix_transform.hpp"

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

struct SceneData {
    // Colors:
    glm::vec3 skyColor;
    glm::vec3 horizontColor;
    glm::vec3 groundColor;
    glm::vec3 color0;
    glm::vec3 color1;
    glm::vec3 color2;

    // Shading:
    glm::vec3 lightDir;
    glm::vec3 lightPower;
    glm::vec3 ambientPower;
    float diffuseIntensity;
    float specularIntensity;
    float ambientIntensity;
    float edgeIntensity;
    float shininess;
    float opacityScale;
    unsigned int shadowStepCount;

    // Fidelity:
    unsigned int rayResolution;
};

SceneData* sceneData;

void allocateAdditionalRenderDataOnDevice()
{
    checkCudaError(cudaMalloc(&sceneData, sizeof(SceneData)));
}

void freeAdditionalRenderDataOnDevice()
{
    checkCudaError(cudaFree(sceneData));
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
__device__ glm::vec3 vectorPower(glm::vec3 v, float n)
{
    float r = glm::length(v);
    float phi = atan2f(v.y, v.x);
    float theta = atan2f(sqrtf(v.x * v.x + v.y * v.y), r);
    float powRN = powf(r, n);
    return glm::vec3(
                powRN * sinf(n * theta) * cosf(n * phi),
                powRN * sinf(n * theta) * sinf(n * phi),
                powRN * cosf(n * theta)
           );
}


__device__ float mandelbulb(
                        glm::vec3 c,
                        float n,
                        unsigned int iterationLimit,
                        float pseudoInfinity
)
{
    c = glm::vec3(c.x, c.z, c.y);    // Rearrange coordinates to change the orietation of the system
    
    glm::vec3 v = glm::vec3{ 0.0f, 0.0f, 0.0f };


    for (unsigned int i = 0; i < iterationLimit; i++) {
        v = vectorPower(v, n) + c;
        if (glm::length(v) > pseudoInfinity) {  // divergent iteration
            break;
        }
    }
    return glm::length(v) / pseudoInfinity;   // Outside the object
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

__device__ float4 sphere(float3 c, float radius)
{
    if (norm3df(c.x, c.y, c.z) < radius)
    {
        return float4(1.0f, 1.0f, 1.0f, 0.01f);
    }
        return float4(0.0f, 0.0f, 0.0f, 0.0f);    
}


/*
    Code source: https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection
*/
__device__ glm::vec2 hit_sphere(const glm::vec3& center, float radius, const glm::vec3& rayStart, const glm::vec3& rayDir){
    glm::vec3 oc = rayStart - center;
    float a = glm::dot(rayDir, rayDir);
    float b = 2.0f * glm::dot(oc, rayDir);
    float c = glm::dot(oc,oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if(discriminant < 0.0f){
        return glm::vec2{ -1.0f, -1.0f };
    }
    else{
        return glm::vec2{(-b - sqrt(discriminant)) / (2.0*a), (-b + sqrt(discriminant)) / (2.0*a) };
    }
}


__device__ glm::vec4 transferColor(
                        float w,
                        float coloringMultiplier,
                        float coloringPower,
                        glm::vec3& color0,
                        glm::vec3& color1,
                        glm::vec3& color2
                     )
{
    float bias = 0.0f;
    float a = powf(1.0f - w + bias, 0.5f);
    if (w - bias > 1.0f) {
        a = 0.0f;
    }
    float t = powf(fminf(fmaxf(w, 0.0f), 1.0f) * coloringMultiplier, coloringPower);
    if (t < 0.5) {
        return glm::vec4(color0 * (1.0f - t * 2.0f) + color1 * t * 2.0f, a);
    }
    else {
        return glm::vec4(color1 * (1.0f - (t - 0.5f) * 2.0f) + color2 * (t - 0.5f) * 2.0f, a);
    }
}

__device__ glm::vec3 tintedAttenuation(glm::vec4& rgba)
{
    float mix = 0.9f;
    return rgba.w * ((1.0f - mix) * glm::vec3(1.0f, 1.0f, 1.0f) + mix * glm::vec3( 1.0f - rgba.x, 1.0f - rgba.y, 1.0f - rgba.z ));
}

__global__ void rayCastMandelbulb(cudaSurfaceObject_t dstSurface, size_t width, size_t height, glm::vec3 eyePos, glm::mat4 rayDirMtx,
                                    float n, unsigned int iterationLimit, float pseudoInfinity,
                                    float coloringMultiplier, float coloringPower,
                                    float rotationRad,
                                    SceneData* scene
                                    )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float ndcX = map(float(x), 0.0f, (float)width - 1.0f, -1.0f, 1.0f);
    float ndcY = map(float(y), 0.0f, (float)height - 1.0f, -1.0f, 1.0f);
    
    glm::vec4 rayDir4 = rayDirMtx * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    glm::vec3 rayDir = glm::normalize(glm::vec3(rayDir4.x, rayDir4.y, rayDir4.z));

    // rotate coordinate system to rotate the bulb:
    glm::mat4 rotationMat(1.0f);
    rotationMat = glm::rotate(rotationMat, rotationRad, glm::vec3(0.0, 1.0, 0.0));
    glm::vec4 v4 = rotationMat * glm::vec4(eyePos, 1.0f); 
    eyePos = glm::vec3(v4.x, v4.y, v4.z);
    v4 = rotationMat * glm::vec4(rayDir, 0.0f);
    rayDir = glm::vec3(v4.x, v4.y, v4.z);

    // We know that the object is in the origin of the coordinate system inside a sqrt(2) sphere:
    glm::vec2 boundingSphereHitDistances = hit_sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.4142f, eyePos, rayDir);
    glm::vec3 backgroundColor = scene->horizontColor * (1.0f - fabsf(rayDir.y)) + scene->skyColor * fmaxf(rayDir.y, 0.0f) + scene->groundColor * -fminf(rayDir.y, 0.0f);   // Create color gradient in background
    if (boundingSphereHitDistances.y < 0.0f) {    // No intersection with the bounding sphere.
        surf2Dwrite(rgbaFloatToInt(float4(backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f)), dstSurface, x * 4, y); // expects byte coordinates. 1 pixel = 4 byte
        return;
    }
    boundingSphereHitDistances.x = fmaxf(boundingSphereHitDistances.x, 0.0f);   // If inside the sphere;
    float maxDistance = boundingSphereHitDistances.y - boundingSphereHitDistances.x;     
    float stepSize = maxDistance / (float)scene->rayResolution;
    glm::vec3 c = eyePos + boundingSphereHitDistances.x * rayDir;
    
    glm::vec3 accumulatedColor = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 accumulatedAttenuation = glm::vec3(0.0f, 0.0f, 0.0f);

    float dx = 0.001f;    // Differentiation step

    for (unsigned int step = 0; step < scene->rayResolution; step++) {
        float sample = mandelbulb(c, n, iterationLimit, pseudoInfinity);
        if (sample < 1.0f) {
            // Approximate gradient:
            float sampleDX = mandelbulb(c + dx * glm::vec3(1.0f, 0.0f, 0.0f), n, iterationLimit, pseudoInfinity);
            float sampleDY = mandelbulb(c + dx * glm::vec3(0.0f, 1.0f, 0.0f), n, iterationLimit, pseudoInfinity);
            float sampleDZ = mandelbulb(c + dx * glm::vec3(0.0f, 0.0f, 1.0f), n, iterationLimit, pseudoInfinity);
            float sampleNDX = mandelbulb(c + dx * glm::vec3(-1.0f, 0.0f, 0.0f), n, iterationLimit, pseudoInfinity);
            float sampleNDY = mandelbulb(c + dx * glm::vec3(0.0f, -1.0f, 0.0f), n, iterationLimit, pseudoInfinity);
            float sampleNDZ = mandelbulb(c + dx * glm::vec3(0.0f, 0.0f, -1.0f), n, iterationLimit, pseudoInfinity);
            glm::vec3 grad = glm::vec3{ sampleDX - sampleNDX, sampleDY - sampleNDY, sampleDZ - sampleNDZ } / dx * 0.5f;
            float gLength = glm::length(grad);
            float avgSample = (sample * 1.25f + sampleDX + sampleDY + sampleDZ + sampleNDX + sampleNDY + sampleNDZ) / 7.25f;
            glm::vec4 rgba = transferColor(avgSample, coloringMultiplier, coloringPower, scene->color0, scene->color1, scene->color2);
            glm::vec3 rgb = glm::vec3(rgba.x, rgba.y, rgba.z);
            glm::vec3 tintedA = tintedAttenuation(rgba);
            glm::vec3 shading = rgb;  // If no gradient is available use the base color as "shaded color"
            
            if (gLength > 0.00001f) {
                glm::vec3 normal = grad / gLength;  // !!! The gradient now is pointing towards the outside of the object. No (-1) !!!
                glm::vec3 halfway = glm::normalize(scene->lightDir - rayDir);
                float shadow = 0.0f;
                glm::vec3 shadowC = c + stepSize * scene->lightDir;
                float shadowStepWeight = 1.0f;
                for (unsigned int j = 0; j < scene->shadowStepCount; j++) {
                    float shadowSample = mandelbulb(shadowC, n, iterationLimit, pseudoInfinity);
                    glm::vec4 shadowRGBA = transferColor(shadowSample, coloringMultiplier, coloringPower, scene->color0, scene->color1, scene->color2);
                    shadowRGBA.w *= scene->opacityScale * shadowStepWeight;
                    shadow += fminf(stepSize * shadowRGBA.w, 1.0f) * (1.0f - shadow);
                    shadowC += stepSize * scene->lightDir;
                    shadowStepWeight *= 0.99f;
                }
                shading = (powf(fmaxf(glm::dot(-rayDir, normal), 0.0f), 3.0f * scene->edgeIntensity) * scene->edgeIntensity + (1.0f - scene->edgeIntensity))  // Edge shading
                    * (
                        + scene->diffuseIntensity * (1.0f - shadow) * scene->lightPower * rgb * fmaxf(glm::dot(scene->lightDir, normal), 0.0f)                                        // Diffuse
                        + scene->specularIntensity * (1.0f - shadow) * scene->lightPower * powf(fmaxf(glm::dot(halfway, normal), 0.0f), scene->shininess)   // Specular
                        + scene->ambientIntensity * scene->ambientPower * rgb                                                                                        // Ambient
                    );
            }
            
            accumulatedColor += stepSize * scene->opacityScale * rgba.w * shading * (1.0f - accumulatedAttenuation);  // Color (under operator)

            glm::vec3 temp = stepSize * scene->opacityScale * tintedA;
            accumulatedAttenuation += glm::vec3(fminf(temp.x, 0.0f), fminf(temp.y, 0.0f), fminf(temp.z, 0.0f)) * (1.0f - accumulatedAttenuation);   // Attenuation (under operator)
            if (accumulatedAttenuation.x > 0.95f && accumulatedAttenuation.y > 0.95f && accumulatedAttenuation.z > 0.95f) {
                accumulatedAttenuation = glm::vec3(1.0f, 1.0f, 1.0f);
                break;
            }
        }
        c += stepSize * rayDir;     // Step along the ray
    }
    float4 outColor = float4{
                        accumulatedColor.x + backgroundColor.x * (1.0f - accumulatedAttenuation.x),
                        accumulatedColor.y + backgroundColor.y * (1.0f - accumulatedAttenuation.y),
                        accumulatedColor.z + backgroundColor.z * (1.0f - accumulatedAttenuation.z),
                        1.0f
                      };
    surf2Dwrite(rgbaFloatToInt(outColor), dstSurface, x * 4, y); // expects byte coordinates. 1 pixel = 4 byte
}


void renderCuda()
{
    if (!theMandelbulbAnimator.getIsHighFidelityHold()) {
        uint32_t nthreads = 16;
        dim3 dimBlock{ nthreads, nthreads };
        dim3 dimGrid{ imageWidth / nthreads + 1, imageHeight / nthreads + 1 };
    
        
        bool isHighFidelity = theMandelbulbAnimator.popIsHighFidelityRender();

        float n = theMandelbulbAnimator.getN();
        unsigned int iterationLimit = (isHighFidelity)? 300 : 8;
        float pseudoInfinity = 16.0f;

        SceneData hostSceneData =  {
            // Colors:
            theMandelbulbAnimator.getSkyColor(),
            theMandelbulbAnimator.getHorizontColor(),
            theMandelbulbAnimator.getGroundColor(),
            theMandelbulbAnimator.getColor0(),
            theMandelbulbAnimator.getColor1(),
            theMandelbulbAnimator.getColor2(),
            // Shading:
            theMandelbulbAnimator.getLightDir(),
            theMandelbulbAnimator.getLightPower(),
            theMandelbulbAnimator.getAmbientPower(),
            theMandelbulbAnimator.getDiffuseIntensity(),
            theMandelbulbAnimator.getSpecularIntensity(),
            theMandelbulbAnimator.getAmbientIntensity(),
            theMandelbulbAnimator.getEdgeIntensity(),
            theMandelbulbAnimator.getShininess(),
            theMandelbulbAnimator.getOpacityScale(),
            (unsigned int)((isHighFidelity)? 10 : 5),
            // Fidelity:
            (unsigned int)((isHighFidelity)? 3000 : 200)
        };

        cudaMemcpy(sceneData, &hostSceneData, sizeof(SceneData), cudaMemcpyHostToDevice);

        rayCastMandelbulb<<<dimGrid, dimBlock>>>(
            surfaceObject,
            imageWidth,
            imageHeight,
            thePerspectiveCamera.eyePos(),
            thePerspectiveCamera.rayDirMatrix((float)imageWidth / (float)imageHeight),
            n,
            iterationLimit,
            pseudoInfinity,
            theMandelbulbAnimator.getColoringMultiplier(),
            theMandelbulbAnimator.getColoringPower(),
            theMandelbulbAnimator.getRotation(),
            sceneData
        );
    }

    //checkCudaError(cudaGetLastError());
    //checkCudaError(cudaDeviceSynchronize()); // not optimal! should be synced with vulkan using semaphores
}