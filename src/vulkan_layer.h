#ifndef VULKAN_INTRO_VULKAN_LAYER_H
#define VULKAN_INTRO_VULKAN_LAYER_H

#define VK_CHECK_RESULT(EXPR) {auto _r = vk::Result(EXPR); if (_r != vk::Result::eSuccess) vulkanLogger.LogFatal("Error in ", __FILE__, " at line ", __LINE__, ": result was ", vk::to_string(_r)); }

struct VulkanImage {
    vk::Image image;
    vk::ImageView view;
    VmaAllocation memory;
    VmaAllocationInfo allocationInfo;
    vk::Format format;
    uint32_t width, height;

    void destroy();
    void* map();
    void unmap();
    void flush();
    void invalidate();
    VulkanImage& name(const char* pName); 
    operator bool() const { return image; }

    void* getNativeWin32Handle();
};

struct VulkanCubeImage {
    vk::Image image;
    vk::ImageView view;
    std::array<vk::ImageView, 6> face;
    VmaAllocation memory;
    VmaAllocationInfo allocationInfo;

    void destroy();
    void* map();
    void unmap();
    void flush();
    void invalidate();
    VulkanCubeImage& name(const char* pName);
    operator bool() const { return image; }
};

struct VulkanBuffer {
    vk::Buffer buffer = nullptr;
    VmaAllocation memory;
    VmaAllocationInfo allocationInfo;

    void destroy();
    void* map();
    void unmap();
    void flush();
    void invalidate();
    VulkanBuffer& name(const char* pName);
    operator bool() const { return buffer; }
};

template <size_t I>
std::array<vk::Buffer, I> extractBufferArray(const std::array<VulkanBuffer, I> buffers) 
{
    std::array<vk::Buffer, I> ret;
    for (size_t i = 0; i < I; ++i) {
        ret[i] = buffers[i].buffer;
    }
    return ret;
}

namespace vma
{
    enum class MemoryUsage
    {
        eUnknown = VMA_MEMORY_USAGE_UNKNOWN,
        eAuto = VMA_MEMORY_USAGE_AUTO,
        eAutoPreferDevice = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        eAutoPreferHost = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        eLazilyAllocated = VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED
    };

    struct AllocationCreateFlagBits
    {
        enum AllocationCreateFlagBitsE
        {
            eDedicatedMemoryBit = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            eNeverAllocateBit = VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT,
            eCreateMappedBit = VMA_ALLOCATION_CREATE_MAPPED_BIT,
            eCanAliasBit = VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT,
            eHostAccessSequentialWriteBit = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            eHostAccessRandomBit = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        };
    };

    using AllocationCreateFlags = VmaAllocationCreateFlags;
}

class VulkanLayer
{
public:
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        std::optional<uint32_t> transferFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
        }
    };

public:
    static VulkanLayer& Instance() { static VulkanLayer instance; return instance; }

    void initializeVulkan();
    void initializeDevices(const vk::SurfaceKHR& surface);

    void destroyVulkan();
    void destroyDevices();

    vk::Result  safeSubmitPresentCommand(vk::PresentInfoKHR presentInfo, bool wait = false);
    vk::Result  safeSubmitGraphicsCommand(vk::SubmitInfo submitInfo, bool wait = false, vk::Fence fence = nullptr);
    vk::Result  safeSubmitTransferCommand(vk::SubmitInfo submitInfo, bool wait = false, vk::Fence fence = nullptr);
    void safeWaitIdle();

    VulkanImage createVolume(uint32_t width, uint32_t height, uint32_t depth, vk::Format format, vk::ImageLayout layout, vk::ImageUsageFlags usage, VmaAllocationCreateInfo memoryInfo, vk::ImageCreateFlags flags = {}, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
    VulkanImage createVolume(uint32_t width, uint32_t height, uint32_t depth, vk::Format format, vk::ImageLayout layout, vk::ImageUsageFlags usage,  vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, vk::ImageCreateFlags flags = {}, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
    VulkanImage createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSample, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, VmaAllocationCreateInfo memoryInfo, vk::ImageTiling tiling = vk::ImageTiling::eOptimal, void* pNext = nullptr);
    VulkanImage createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSample, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, vk::ImageTiling tiling = vk::ImageTiling::eOptimal, void* pNext = nullptr);
    VulkanCubeImage createCubeMap(uint32_t width, uint32_t height, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, VmaAllocationCreateInfo memoryInfo, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
    VulkanCubeImage createCubeMap(uint32_t width, uint32_t height, vk::Format format, vk::ImageLayout layout, vk::ImageAspectFlags aspect, vk::ImageUsageFlags usage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
    VulkanBuffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, VmaAllocationCreateInfo memoryInfo);
    VulkanBuffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {});
    VulkanBuffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vma::AllocationCreateFlags allocationFlags);

    VulkanBuffer requestStagingBuffer(vk::DeviceSize size);
    void returnStagingBuffer(VulkanBuffer& buffer);

    vk::PipelineLayout createPipelineLayout(std::initializer_list<vk::DescriptorSetLayout> descriptorSets, uint32_t pushConstantRange = 0, vk::ShaderStageFlags pushConstantFlags = {});
    vk::Pipeline createComputePipeline(vk::ShaderModule computeShader, vk::PipelineLayout pipelineLayout);
    vk::Pipeline createGraphicsPipeline(vk::GraphicsPipelineCreateInfo createInfo);
    vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);
    void generateMipmaps(vk::Image image, vk::Format imageFormat, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels);

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height, uint32_t depth = 1u);

    vk::CommandBuffer beginSingleTimeCommands();
    vk::CommandBuffer beginSingleTimeTransferCommand();
    void endSingleTimeTransferCommand(vk::CommandBuffer commandBuffer);
    void endSingleTimeCommands(vk::CommandBuffer commandBuffer);

    void* getNativeWin32Handle(vk::Semaphore semaphore);

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    vk::Format findDepthFormat();
    
    vk::PipelineShaderStageCreateInfo loadShaderStageCreateInfo(const std::string& path, vk::ShaderStageFlagBits stage);
    vk::ShaderModule createShaderModule(const std::vector<char>& code);

    static std::vector<char> readFile(const std::string& filename);
    static std::vector<char> readFile(const std::filesystem::path& file);
    static void writeFile(const std::string& filename, const std::vector<uint8_t>& data);
    static void writeFile(const std::filesystem::path& file, const std::vector<uint8_t>& data);

    VulkanImage name(VulkanImage image, const char* name)
    {
        name_impl(image.image.objectType, (uint64_t)(typename vk::Image::CType)image.image, name);
        return image;
    }

    template <typename T>
    T name(T handle, const std::string& mName)
    {
        return name(handle.objectType, (uint64_t)(typename T::CType)handle, mName.c_str());
    }

    template <typename T>
    T name(T handle, const char* name)
    {
        name_impl(handle.objectType, (uint64_t)(typename T::CType)handle, name);
        return handle;
    }

private:
    void name_impl(vk::ObjectType objectType, uint64_t handle, const char* pObjectName);

private:
    VulkanLayer() = default;

private:
    void createInstance(bool enableValidationLayers);
    void createDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevices();
    void createCommandPool();
    void createPipelineCache();
    void writePipelineCache();

private:
    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device);
    void collectRequiredInstanceExtensions(bool enableValidationLayers);
    void collectDeviceExtensions();
    bool checkRequiredExtensionSupport();
    bool checkValidationLayerSupport();
    bool isDeviceSuitable(const vk::PhysicalDevice& device);
    vk::SampleCountFlagBits getMaxUsableSampleCount();
    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device);

private:
    std::vector<const char*> validationLayers;
    std::vector<const char*> instanceExtensions;
    std::vector<const char*> deviceExtensions;

public:
    vk::Instance instance;
    vk::DebugUtilsMessengerEXT messenger;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    VmaAllocator allocator;
    vk::Queue graphicsQueue, presentQueue, transferQueue;
    uint32_t graphicsFamilyIndex;
    vk::CommandPool commandPool, transientCommandPool, transferCommandPool;
    vk::PipelineCache pipelineCache;
    vk::SampleCountFlagBits maxMSAASamples = vk::SampleCountFlagBits::e1;
    vk::PhysicalDeviceProperties physicalDeviceProperties;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties;
    vk::PhysicalDeviceSampleLocationsPropertiesEXT sampleLocationProperties;    

    float timestampPeriod;
    bool isValidationEnabled;

    QueueFamilyIndices queueFamilyIndices;

    mutable std::mutex mQueueMutex; // TODO: seperate mutex for graphics and transfer queues if available

public:
    bool swapChainMutableFormatAvailable = false, memoryBudgetAvailable = false;

private:
    const vk::SurfaceKHR* pSurface;
};

inline auto& theVulkanLayer = VulkanLayer::Instance();

#endif//VULKAN_INTRO_VULKAN_LAYER_H
