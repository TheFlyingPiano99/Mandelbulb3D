#version 450

layout(location = 0) out vec2 uv;

void main()
{
    // three vertices, no vertex attributes, triangle rasterization
    // gl_VertexIndex goes from 0 to 2
    // 0x00: uv = 0, 0
    // 0x01: uv = 2, 0
    // 0x10: uv = 0, 2
    // |\ 
    // | \ 
    // |__\ 
    // 0, 0: gl_Position = -1, -1
    // 2, 0: gl_Position = +3, -1
    // 0, 2: gl_Position = -1, +3
    // oversized traingle covering full NDC -> clipping
    // gets clipped at half points -> uv will be in 0 - 1 range
    uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
}