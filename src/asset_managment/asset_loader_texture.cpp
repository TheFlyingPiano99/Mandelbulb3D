#include "asset_loader_texture.h"
#include "asset_manager.h"
#include <stbi/stb_image.h>

TextureHandler handleTexture(AssetManager& am, const std::string& texName, const std::string& /* colorSpace */, bool sRGB, bool mipMapped)
{
    int mWidth, mHeight, mChannels;
    //stbi_set_flip_vertically_on_load(true);
    unsigned char *image = stbi_load(texName.c_str(), &mWidth, &mHeight, &mChannels, STBI_rgb_alpha);
    if (!image) {
        logger.LogError("Texture file is invalid " + texName);
        return am.add(Texture::empty(texName));
    }

    auto width = (uint32_t)mWidth, height = (uint32_t)mHeight;

    ImageResource img;
    img.width = width;
    img.height = height;
    img.channel = 4;
    img.componentSize = 1;
    img.sRGB = sRGB;

    if (mipMapped) {
        img.mipLevel = static_cast<uint32_t>(std::floor(std::log2(std::max(img.width, img.height)))) + 1;
    } else {
        img.mipLevel = 1;
    }
   
    img.mLocation = ResourceLocation::InMemoryBit;

    img.data.resize(width * height * 4);
    memcpy(img.data.data(), image, width * height * 4);
    stbi_image_free(image);

    if (sRGB) {
        img.pixelFormat = vk::Format::eR8G8B8A8Srgb;
    } else {
        img.pixelFormat = vk::Format::eR8G8B8A8Unorm; 
    }

    img.usage = vk::ImageUsageFlagBits::eSampled;

    img.upload();
    auto ih = theResourceManager.addImage(std::move(img));

    auto th = Texture::empty(texName);
    th.imageHandler = ih;
    th.samplerHandler = theResourceManager.defaultSampler;

    return am.add(std::move(th));
}

TextureHandler handleTextureCombination(AssetManager& am, const std::vector<std::pair<std::string, std::string>>& textures, bool mipMapped)
{
    auto _l1 = logger.PushContext("Combining texture:");

    std::vector<stbi_uc*> d(textures.size(), nullptr);
    std::vector<glm::uvec3> dim(textures.size());

    uint32_t channel = (uint32_t)textures.size();
    for (size_t i = 0; i < textures.size(); ++i) {
        glm::ivec3 cdim;
        d[i] = stbi_load(textures[i].first.c_str(), &cdim.x, &cdim.y, &cdim.z, 1); // TODO: handle bad width height
        dim[i] = (glm::uvec3)cdim;
        if (!d[i]) {
            for (auto& ptr : d) {
                if (ptr) stbi_image_free(ptr);
            }
            logger.LogError(std::string{"failed to read texture image: "} + textures[i].first);
            return theAssetManager.add(Texture::empty("comb_temp_name")); // TODO: name
        }
    }

    // workaround for 3 channel textures
    if (textures.size() == 3) {
        d.push_back((stbi_uc*)calloc(dim[0].x * dim[0].y, sizeof(stbi_uc)));
        dim.emplace_back(dim[0].x, dim[0].y, 1);
        channel = 4;
    }

    ImageResource img;
    img.width = dim[0].x;
    img.height = dim[0].y;
    img.channel = channel;
    img.componentSize = 1;
    img.sRGB = false;

    if (mipMapped) {
        img.mipLevel = static_cast<uint32_t>(std::floor(std::log2(std::max(img.width, img.height)))) + 1;
    } else {
        img.mipLevel = 1;
    }
   
    img.mLocation = ResourceLocation::InMemoryBit;

    img.data.resize(img.width * img.height * channel);

    int actC = 0;
    for (size_t N = 0; N < d.size(); ++N) {
        for (uint32_t i = 0; i < dim[N].x * dim[N].y; ++i) {
            std::byte* fromOffset = (std::byte*)d[N] + i * 1;
            std::byte* toOffset = img.data.data() + i * channel;
            for (int c = 0; c < 1; ++c) {
                toOffset[actC] = fromOffset[c];
            }
        }
        actC += 1;
    }

    for (auto& ptr : d) {
        if (ptr) stbi_image_free(ptr);
    }

    if (channel == 4) {
        img.pixelFormat = vk::Format::eR8G8B8A8Unorm;
    } else assert(false && "unhandled channel format.");

    img.usage = vk::ImageUsageFlagBits::eSampled;

    img.upload();
    auto ih = theResourceManager.addImage(std::move(img));

    auto th = Texture::empty("comb_texture_temp_name"); // TODO: name
    th.imageHandler = ih;
    th.samplerHandler = theResourceManager.defaultSampler;

    return am.add(std::move(th));
}
