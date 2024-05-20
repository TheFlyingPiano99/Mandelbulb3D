#version 450

#include "forward_common.glsl"

// vertex inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

// uniform data
layout(set = 2, binding = 0) uniform UniformBufferObject {
    PerObjectUniformBufferObject perObject;
};

// vertex shader outputs == fragment shader inputs
layout(location = 0) out vec3 viewSpacePosition;
layout(location = 1) out vec3 viewSpaceNormal;
layout(location = 2) out vec2 texCoord;

void main() {
    viewSpacePosition = (perObject.mv * vec4(inPosition, 1)).xyz;
    viewSpaceNormal = (perObject.normal * vec4(inNormal, 0)).xyz;
    texCoord = inTexCoord;
    gl_Position = perObject.mvp * vec4(inPosition, 1.0);
}