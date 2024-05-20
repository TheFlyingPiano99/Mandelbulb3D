#include "material_traits.h"
#include "utility/utility.hpp"

using namespace std;

size_t hash<vk::SamplerCreateInfo>::operator()(const vk::SamplerCreateInfo &x) const
{
    size_t seed = 42607514;
    hash_combine(seed, (int)x.addressModeU, (int)x.addressModeV, (int)x.addressModeW, (int)x.borderColor,
                 (int)x.anisotropyEnable, (int)x.maxAnisotropy, (unsigned int)x.flags, (int)x.magFilter,
                 (int)x.minFilter, (int)x.mipmapMode, (int)x.mipLodBias, (int)x.compareEnable, (int)x.compareOp,
                 (int)x.minLod, (int)x.maxLod, (int)x.unnormalizedCoordinates);
    return seed;
}
