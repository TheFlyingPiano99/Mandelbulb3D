#include "shader_manager.h"
#include "vulkan_layer.h"
#include "utility/utility.hpp"
#include "constants.h"

namespace fs = std::filesystem;

void ShaderManager::initialize()
{
    rebuildShaders();
}

ShaderManager& ShaderManager::instance()
{
    static ShaderManager sm;
    return sm;
}

std::set<ShaderDependencyReference> ShaderManager::buildDependencyGraph()
{
    std::set<ShaderFileData> newShaderFiles;

    for (auto& p : fs::recursive_directory_iterator(cDirShaderSources)) {
        if (p.is_regular_file()) {
            const auto& ext = p.path().extension().string();
            if (ext == ".frag" || ext == ".vert" || ext == ".geom" || 
                ext == ".tesc" || ext == ".tese" || ext == ".comp" ||
                ext == ".rchit" || ext == ".rgen" || ext == ".rmiss" || 
                ext == ".rcall" || ext == ".rint" || ext == ".rahit" || 
                ext == ".task" || ext == ".mesh" || ext == ".glsl")
            {
                ShaderFileData data;
                data.lastWriteTime = p.last_write_time();
                data.path = p.path().string();
                data.rawContent = readFile(data.path.c_str());
                data.name = p.path().filename().string();
                data.extension = ext;

                if (ext == ".frag") data.stage = vk::ShaderStageFlagBits::eFragment;
                else if (ext == ".rahit") data.stage = vk::ShaderStageFlagBits::eAnyHitKHR;
                else if (ext == ".vert") data.stage = vk::ShaderStageFlagBits::eVertex;
                else if (ext == ".geom") data.stage = vk::ShaderStageFlagBits::eGeometry;
                else if (ext == ".tesc") data.stage = vk::ShaderStageFlagBits::eTessellationControl;
                else if (ext == ".tese") data.stage = vk::ShaderStageFlagBits::eTessellationEvaluation;
                else if (ext == ".comp") data.stage = vk::ShaderStageFlagBits::eCompute;
                else if (ext == ".rchit") data.stage = vk::ShaderStageFlagBits::eClosestHitKHR;
                else if (ext == ".rgen") data.stage = vk::ShaderStageFlagBits::eRaygenKHR;
                else if (ext == ".rmiss") data.stage = vk::ShaderStageFlagBits::eMissKHR;
                else if (ext == ".rcall") data.stage = vk::ShaderStageFlagBits::eCallableKHR;
                else if (ext == ".rint") data.stage = vk::ShaderStageFlagBits::eIntersectionKHR;
                else if (ext == ".task") data.stage = vk::ShaderStageFlagBits::eTaskEXT;
                else if (ext == ".mesh") data.stage = vk::ShaderStageFlagBits::eMeshEXT;
                else data.stage = std::nullopt;

                // find include dependencies
                std::stringstream ss(data.rawContent);
                std::string tmp;
                while (std::getline(ss, tmp, '\n'))
                {
                    trim(tmp);
                    std::string inc = "#include";
                    if (tmp.size() > inc.size()) {
                        auto res = std::mismatch(inc.begin(), inc.end(), std::begin(tmp), tmp.end());
                        if (res.first == inc.end()) {
                            tmp = tmp.substr((size_t)(res.second - tmp.begin()));
                            tmp.erase(std::remove(tmp.begin(), tmp.end(), '"'), tmp.end());
                            trim(tmp);
                            auto tempp = fs::path{ tmp };
                            data.dependencies.push_back(tempp.filename().string());
                        }
                    }
                }
                if (newShaderFiles.contains(data)) {
                    logger.LogError("Found shader file with the same name: ", data.name, ". Path1: ", data.path, " ; Path2: ", newShaderFiles.find(data)->path);
                } else {
                    newShaderFiles.insert(std::move(data));
                }
            }
        }
    }

    // linearize recursive dependencies
    std::function<std::set<std::string>(ShaderFileData&)> collectDependencies;
    collectDependencies = [&](ShaderFileData& sfd) {
        std::set<std::string> deps{ sfd.dependencies.begin(), sfd.dependencies.end() };
        for (auto& d : sfd.dependencies) {
            ShaderFileData temp; temp.name = d;
            auto found = newShaderFiles.find(temp);
            if (found != newShaderFiles.end()) {
                auto ret = collectDependencies(const_cast<ShaderFileData&>(*found)); // we do not modify the name which is the key
                deps.insert(ret.begin(), ret.end());
            }
        }
        return deps;
    };
    for (auto& cit : newShaderFiles) {
        auto& it = const_cast<ShaderFileData&>(cit); // we do not modify the name which is the key
        auto deps = collectDependencies(it);
        it.dependencies = std::vector<std::string>{ deps.begin(), deps.end() };
    }

    // calculate dependants relation
    for (auto& cit : newShaderFiles) {
        for (auto& d : cit.dependencies) {
            ShaderFileData temp; temp.name = d;
            auto found = newShaderFiles.find(temp);
            if (found != newShaderFiles.end()) {
                (const_cast<std::vector<std::string>&>(found->dependants)).push_back(cit.name);
            }
        }
    }

    // merge into existing data structure
    std::set<ShaderDependencyReference> references;
    for (auto& cit : newShaderFiles) {
        auto found = shaderFiles.find(cit);
        if (found == shaderFiles.end()) { // new file
            shaderFiles.insert(cit);
        } else if (found->lastWriteTime < cit.lastWriteTime) { // existing but modified file
            std::vector<std::string> v = cit.dependants;
            v.push_back(found->name);
            for (auto& str : v) {
                for (auto& s : shaderDependencies) {
                    if (std::find(s->shaders.begin(), s->shaders.end(), str) != s->shaders.end()) {
                        references.insert(s.get()); // collect all uniqe dependencies
                    }
                }
            }
            const_cast<ShaderFileData&>(*found) = cit; // replace file with new one
        }
    }
    return references;
}

void ShaderManager::compileShaders(bool all)
{
    fs::create_directories(cDirShader);
    std::set<fs::path> compilePaths, dummyPaths;

    for (auto& cit : shaderFiles) {
        auto strPath = cDirShader / (cit.name + ".spv");
        auto outPath = fs::path{ strPath };
        if (all || !fs::exists(outPath) || cit.lastWriteTime > fs::last_write_time(outPath)) {
            if (cit.compilable()) {
                compilePaths.insert(cit.path);
            } else {
                dummyPaths.insert(cit.path);
            }
            for (auto& d : cit.dependants) {
                ShaderFileData temp; temp.name = d;
                auto found = shaderFiles.find(temp);
                if (found != shaderFiles.end() && found->compilable()) {
                    compilePaths.insert(found->path);
                }
            }
        }
    }
    
    for (auto& p : compilePaths) {
        auto outPath = cDirShader / (p.filename().string() + ".spv");
        vulkanLogger.LogInfo("Compiling shader: " + p.filename().string());
#ifdef __unix__
        system(("glslc -c --target-spv=spv1.5 \"" + p.string() + "\" -o \"" + outPath.string() + '\"').c_str());
#elif defined(_WIN32) || defined(_WIN64)
        system(("%VULKAN_SDK%/Bin/glslc.exe -c -g --target-env=vulkan1.3 --target-spv=spv1.6  \"" + p.string() + "\" -o \"" + outPath.string() + '\"').c_str());
        //system(("%VULKAN_SDK%/Bin/glslangValidator.exe -e main -gVs -V --target-env vulkan1.3 \"" + p.string() + "\" -o \"" + outPath.string() + '\"').c_str());
#endif
    }

    for (auto& p : dummyPaths) {
        auto outPath = cDirShader / (p.filename().string() + ".spv");
        vulkanLogger.LogInfo("Compiling shader: " + p.filename().string());
        fs::copy_file(p, outPath, fs::copy_options::overwrite_existing);
    }
}

ShaderFileData& ShaderManager::find(const std::string& name)
{
    ShaderFileData temp; temp.name = name;
    auto found = shaderFiles.find(temp);
    assert(found != shaderFiles.end());
    return const_cast<ShaderFileData&>(*found);
}

void ShaderManager::rebuildShaders(bool all)
{
    auto references = buildDependencyGraph();
    compileShaders(all);
    // execute dependencies
    for (auto& it : references) {
        it->call();
    }
}

ShaderBuilder ShaderManager::getShader(const std::string& name)
{
    return ShaderBuilder{ find(name) };
}

void ShaderManager::destroyShaders(vk::ArrayProxy<vk::PipelineShaderStageCreateInfo> shaders)
{
    for (auto& it : shaders) {
        theVulkanLayer.device.destroy(it.module);
    }
}

ShaderDependencyReference ShaderManager::addAndExecuteShaderDependency(std::initializer_list<std::string> shaders, std::function<void(vk::ArrayProxy<ShaderBuilder>)> handler)
{
    auto trueHandler = [handler = std::move(handler), this](const std::vector<std::string>& shaders) {
        std::vector<ShaderBuilder> ss;
        ss.reserve(shaders.size());
        for (auto& it : shaders) {
            ss.push_back(ShaderBuilder{ find(it) });
        }
        handler(ss);
    };
    shaderDependencies.push_back(std::make_unique<ShaderDependency>(std::vector<std::string>{ std::begin(shaders), std::end(shaders) }, std::move(trueHandler)));
    shaderDependencies.back()->call();
    return shaderDependencies.back().get();
}

ShaderDependencyReference ShaderManager::addAndExecuteShaderDependency(const std::vector<std::string>& shaders, std::function<void(vk::ArrayProxy<ShaderBuilder>)> handler)
{
    auto trueHandler = [handler = std::move(handler), this](const std::vector<std::string>& shaders) {
        std::vector<ShaderBuilder> ss;
        ss.reserve(shaders.size());
        for (auto& it : shaders) {
            ss.push_back(ShaderBuilder{ find(it) });
        }
        handler(ss);
    };
    shaderDependencies.push_back(std::make_unique<ShaderDependency>(shaders, std::move(trueHandler)));
    shaderDependencies.back()->call();
    return shaderDependencies.back().get();
}

ShaderDependencyReference ShaderManager::addAndExecuteComputeShaderDependency(const std::string& shader, vk::PipelineLayout pipelineLayout, vk::Pipeline* pPipeline, const char* name)
{
    return addAndExecuteShaderDependency({ shader }, [pPipeline, pipelineLayout, name](auto shaders) {
        vk::PipelineShaderStageCreateInfo shader = shaders.begin()[0].build();
        if (*pPipeline) theVulkanLayer.device.destroy(*pPipeline);
        *pPipeline = theVulkanLayer.name(theVulkanLayer.createComputePipeline(shader.module, pipelineLayout), name);
        theShaderManager.destroyShaders(shader);
    });
}

void ShaderManager::removeShaderDependency(ShaderDependencyReference ref)
{
    std::erase_if(shaderDependencies, [&](const auto& it) { return it.get() == ref; });
}

void ShaderManager::destroyPipelines(std::initializer_list<std::pair<vk::Pipeline&, ShaderDependencyReference&>> list)
{
    for (auto& it : list) {
        theVulkanLayer.device.destroy(it.first);
        it.first = nullptr;
        removeShaderDependency(it.second);
    }
}

vk::PipelineShaderStageCreateInfo ShaderBuilder::build() const
{
    auto path = cDirShader / (shader.name + ".spv");
    return theVulkanLayer.loadShaderStageCreateInfo(path.string(), shader.stage.value());
}
