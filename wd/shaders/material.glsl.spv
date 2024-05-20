#extension GL_EXT_nonuniform_qualifier : enable

#include "utility.glsl"

layout(set = 1, binding = 0) buffer Materials {
    float entries[];
} mats;
layout(set = 1, binding = 1) uniform sampler2D textures[];

struct GLTFMaterial {
    vec3 diffuse;
    float opacity;
    vec3 normal;
    float metallic, roughness;
};

GLTFMaterial readSimpleMaterial(int materialIndex)
{
    GLTFMaterial ret;
    ret.diffuse = vec3(
                    mats.entries[materialIndex + 3],
                    mats.entries[materialIndex + 4],
                    mats.entries[materialIndex + 5]);
    ret.metallic = mats.entries[materialIndex + 7];
    ret.roughness = mats.entries[materialIndex + 8];
    ret.opacity = mats.entries[materialIndex + 6];
    return ret;
}

GLTFMaterial readMaterial(int materialIndex, vec2 texCoord)
{
    GLTFMaterial ret;

    float diffuseTextureID = mats.entries[materialIndex + 0];
    if (diffuseTextureID < 0.0) {
        ret.diffuse = vec3(1.0);
        ret.opacity = 1.0;
    } else {
        vec4 data = texture(textures[nonuniformEXT(int(diffuseTextureID))], texCoord);
        ret.diffuse = data.rgb;
        ret.opacity = data.a;
    }
    ret.diffuse *= vec3(
                    mats.entries[materialIndex + 3],
                    mats.entries[materialIndex + 4],
                    mats.entries[materialIndex + 5]);
    ret.opacity *=  mats.entries[materialIndex + 6];

    float metallicRoughnessTextureID = mats.entries[materialIndex + 2];
    if (metallicRoughnessTextureID < 0.0) {
        ret.metallic = ret.roughness = 1.0;
    } else {
        vec2 temp = texture(textures[nonuniformEXT(int(metallicRoughnessTextureID))], texCoord).xy;
        ret.metallic = temp.x;
        ret.roughness = temp.y;
    }
    ret.metallic *= mats.entries[materialIndex + 7];
    ret.roughness *= mats.entries[materialIndex + 8];

    return ret;
}

float readOpacity(int materialIndex, vec2 texCoord)
{
    float opacity;
    float diffuseTextureID = mats.entries[materialIndex + 0];
    if (diffuseTextureID < 0.0) {
        opacity = 1.0;
    } else {
        vec4 data = texture(textures[nonuniformEXT(int(diffuseTextureID))], texCoord);
        opacity = data.a;
    }
    opacity *=  mats.entries[materialIndex + 6];
    return opacity;
}
