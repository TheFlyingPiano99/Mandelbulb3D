#ifndef VULKAN_INTRO_SHADER_MANAGER_H
#define VULKAN_INTRO_SHADER_MANAGER_H

struct ColouredMesh;

struct ShaderFileData {
    std::string name, path, rawContent, extension;
    std::vector<std::string> dependencies, dependants;
    std::filesystem::file_time_type lastWriteTime;
    std::optional<vk::ShaderStageFlagBits> stage;

    bool compilable() const {
        return extension == ".frag" || extension == ".vert" || extension == ".geom" ||
            extension == ".tesc" || extension == ".tese" || extension == ".comp" ||
            extension == ".rchit" || extension == ".rgen" || extension == ".rmiss" ||
            extension == ".rcall" || extension == ".rint" || extension == ".rahit" ||
            extension == ".mesh" || extension == ".task";
    }
    bool operator<(const ShaderFileData& that) const { return name < that.name; }
};

class ShaderBuilder {
public:
    ShaderBuilder(const ShaderFileData& shader) : shader{ shader } {}
    vk::PipelineShaderStageCreateInfo build() const;

private:
    const ShaderFileData& shader;
};

struct ShaderDependency {
    std::vector<std::string> shaders;
    std::function<void(const std::vector<std::string>&)> handler;

    void call() { handler(shaders); }
};

using ShaderDependencyReference = ShaderDependency*;

class ShaderManager {
public:
    static ShaderManager& instance();    

    void initialize();

    void rebuildShaders(bool all = false);
    ShaderBuilder getShader(const std::string& name);
    void destroyShaders(vk::ArrayProxy<vk::PipelineShaderStageCreateInfo> shaders);
    ShaderDependencyReference addAndExecuteShaderDependency(std::initializer_list<std::string> shaders, std::function<void(vk::ArrayProxy<ShaderBuilder> shaders)> handler);
    ShaderDependencyReference addAndExecuteShaderDependency(const std::vector<std::string>& shaders, std::function<void(vk::ArrayProxy<ShaderBuilder> shaders)> handler);
    ShaderDependencyReference addAndExecuteComputeShaderDependency(const std::string& shader, vk::PipelineLayout pipelineLayout, vk::Pipeline* pPipeline, const char* name);
    void removeShaderDependency(ShaderDependencyReference ref);

    void destroyPipelines(std::initializer_list<std::pair<vk::Pipeline&, ShaderDependencyReference&>> list);

private:
    ShaderManager() = default;
    using FnCreatePayloadData = void(*)(void* pData, ColouredMesh& pMesh);
    std::set<ShaderDependencyReference> buildDependencyGraph();
    void compileShaders(bool all = false);
    ShaderFileData& find(const std::string& name);

private:
    std::set<ShaderFileData> shaderFiles;
    std::vector<std::unique_ptr<ShaderDependency>> shaderDependencies;
};

inline auto& theShaderManager = ShaderManager::instance();

#endif//VULKAN_INTRO_SHADER_MANAGER_H
