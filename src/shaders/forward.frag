#version 450

#include "forward_common.glsl"
#include "utility.glsl"
#include "material.glsl"
#include "shading.glsl"

layout(location = 0) in vec3 viewSpacePosition;
layout(location = 1) in vec3 viewSpaceNormal;
layout(location = 2) in vec2 texCoord;

layout(set = 0, binding = 0) uniform PerFrameUBO {
    PerFrameUniformBufferObject perFrame;
};

const int MAX_POINTLIGHT_COUNT = 50;
layout(set = 0, binding = 1) uniform PointLightsUBO {
    ivec4 lightCount_pad3;
    PointLight pointLights[MAX_POINTLIGHT_COUNT];
} plubo;

layout(set = 2, binding = 0) uniform PerObjectUBO {
    PerObjectUniformBufferObject perObject;
};

layout(location = 0) out vec4 outColor;

vec4 shade(inout GLTFMaterial mat) {
    vec3 finalColor = vec3(0);

    for (uint i = 0; i < plubo.lightCount_pad3.x; ++i) {
        vec3 lpos = (perFrame.v * plubo.pointLights[i].pos).xyz;
        vec3 lpower = plubo.pointLights[i].power.rgb * plubo.pointLights[i].power.a;
        vec3 irradiance = calculateIrradiance(viewSpacePosition, lpos, lpower);
        finalColor += viewSpaceShading(viewSpacePosition, mat.diffuse, mat.normal, mat.metallic, mat.roughness, lpos - viewSpacePosition, irradiance).xyz;
    }

    finalColor += mat.diffuse * perFrame.ambientLight.rgb * perFrame.ambientLight.a;

    finalColor = tonemapACES(finalColor);

    return vec4(finalColor, 1.0);
}

void main() {
    GLTFMaterial material = readMaterial(perObject.materialIndex, texCoord);
    material.normal = normalize(viewSpaceNormal);
    outColor = shade(material);
}