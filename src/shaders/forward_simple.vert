#version 450

#include "forward_common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 2, binding = 0) uniform UniformBufferObject {
    PerObjectUniformBufferObject perObject;
};

layout(location = 0) out vec3 viewSpacePosition;
layout(location = 1) out vec3 viewSpaceNormal;

void main() {
    viewSpacePosition = (perObject.mv * vec4(inPosition, 1)).xyz;
    viewSpaceNormal = (perObject.normal * vec4(inNormal, 0)).xyz;
    gl_Position = perObject.mvp * vec4(inPosition, 1.0);
}