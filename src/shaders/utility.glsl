#ifndef UTILITY_GLSL
#define UTILITY_GLSL

struct Vertex {
    vec3 p;
    vec3 n;
    vec2 uv;
};

float ray_plane_intersection(vec3 ray_o, vec3 ray_d, vec3 plane_n, float plane_d) {
    return (-plane_d - dot(plane_n, ray_o)) / dot(plane_n, ray_d);
}

vec3 encodeSRGB(vec3 linearRGB)
{
    vec3 a = 12.92 * linearRGB;
    vec3 b = 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055;
    vec3 c = step(vec3(0.0031308), linearRGB);
    return mix(a, b, c);
}

vec3 decodeSRGB(vec3 screenRGB)
{
    vec3 a = screenRGB / 12.92;
    vec3 b = pow((screenRGB + 0.055) / 1.055, vec3(2.4));
    vec3 c = step(vec3(0.04045), screenRGB);
    return mix(a, b, c);
}

vec2 encodeNormal(vec3 n)
{
    float p = sqrt(n.z*8+8);
    return n.xy/p + 0.5;
}

vec3 decodeNormal(vec2 enc)
{
    vec2 fenc = enc*4-2;
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}

float signNotZero(in float k) {
    return (k >= 0.0) ? 1.0 : -1.0;
}

vec2 signNotZero(in vec2 v) {
    return vec2(signNotZero(v.x), signNotZero(v.y));
}

/** Assumes that v is a unit vector. The result is an octahedral vector on the [-1, +1] square. */
vec2 encodeNormal2(in vec3 v) {
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    vec2 result = v.xy * (1.0 / l1norm);
    if (v.z < 0.0) {
        result = (1.0 - abs(result.yx)) * signNotZero(result.xy);
    }
    return result;
}


/** Returns a unit vector. Argument o is an octahedral vector packed via octEncode,
    on the [-1, +1] square*/
vec3 decodeNormal2(vec2 o) {
    vec3 v = vec3(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
    if (v.z < 0.0) {
        v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    }
    return normalize(v);
}

float rand(vec2 co)
{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// All components are in the range [0-1], including hue.
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// All components are in the range [0-1], including hue.
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 rgb2yuv(vec3 rgb)
{
	vec3 yuv = vec3(0.0);

	yuv.x = rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114;
	yuv.y = rgb.r * -0.169 + rgb.g * -0.331 + rgb.b * 0.5 + 0.5;
	yuv.z = rgb.r * 0.5 + rgb.g * -0.419 + rgb.b * -0.081 + 0.5;

	return yuv;
}

vec3 yuv2rgb(vec3 yuv)
{
	yuv.x=1.1643*(yuv.x-0.0625);
	yuv.y=yuv.y-0.5;
	yuv.z=yuv.z-0.5;

	vec3 rgb = vec3(0.0);

	rgb.r=yuv.x+1.5958*yuv.z;
	rgb.g=yuv.x-0.39173*yuv.y-0.81290*yuv.z;
	rgb.b=yuv.x+2.017*yuv.y;

	return rgb;
}

// depthSample from depthTexture.r, for instance using vulkan convention z:[0, 1]
float linearDepth(float depthSample, float nearPlane, float farPlane)
{
    return nearPlane * farPlane / (farPlane - depthSample * (farPlane - nearPlane));
}

// result suitable for assigning to gl_FragDepth using vulkan convention z:[0, 1]
float depthSample(float linearDepth, float nearPlane, float farPlane)
{
    return (farPlane - nearPlane * farPlane / linearDepth) / (farPlane - nearPlane);
}

// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-16-accurate-atmospheric-scattering
float phaseFunction1(float cosphi, float g)
{
    float gg = g * g;
    return (3.0 * (1.0 - gg)) / (2.0 * (2 + gg)) * (1 + cosphi * cosphi) / pow(1 + gg - 2 * g * cosphi, 1.5);
}

#endif // UTILITY_GLSL
