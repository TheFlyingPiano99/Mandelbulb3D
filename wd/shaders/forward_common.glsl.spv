#ifndef FORWARD_COMMON_GLSL
#define FORWARD_COMMON_GLSL

struct PerFrameUniformBufferObject {
    mat4 v;
    vec4 ambientLight;
};

struct PerObjectUniformBufferObject {
    mat4 mv;
    mat4 mvp;
    mat4 normal;
    int materialIndex;
};

struct PointLight {
    vec4 pos, power;
};

#endif//FORWARD_COMMON_GLSL