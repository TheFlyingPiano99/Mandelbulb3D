const float PI = 3.14159265359;

float clampedDot(vec3 v1, vec3 v2)
{
    return max(dot(v1, v2), 0.0);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float trowbridgeReitzMicrofacetDistribution(float alpha, vec3 N, vec3 H)
{
    float alpha2 = alpha * alpha;
    float NdotH = clampedDot(N, H);
    float temp = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * temp * temp);
}

float smithJointGGX(vec3 N, vec3 V, vec3 L, float alpha)
{
    float NdotV = clampedDot(N, V);
    float NdotL = clampedDot(N, L);
    float alpha2 = alpha * alpha;

    float t1 = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha2) + alpha2);
    float t2 = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha2) + alpha2);

    return 0.5 / (t1 + t2);
}

vec3 calculateIrradiance(vec3 position, vec3 lightPosition, vec3 power)
{
    vec3 dv = lightPosition - position;
    float dist2 = dot(dv, dv);

    float attenuation = 1.0 / dist2;
    return power * attenuation;
}

vec4 anySpaceShading(vec3 viewDirection, vec3 albedo, vec3 N, float metallic, float roughness, vec3 lightDirection, vec3 irradiance)
{
    vec3 V = viewDirection;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float alpha = roughness * roughness;
    vec3 cdiff = mix(albedo * (1.0 - 0.04), vec3(0.0), metallic);

    vec3 L = lightDirection;
    vec3 H = normalize(L + V);

    vec3 F = fresnelSchlick(clampedDot(H, V), F0);
    float D = trowbridgeReitzMicrofacetDistribution(alpha, N, H);
    float vis = smithJointGGX(N, V, L, alpha);

    vec3 specular = F * vis * D;
    vec3 diffuse = (1.0 - F) * cdiff / PI;

    float NdotL = clampedDot(N, L);
    return vec4(max(vec3(0),(diffuse + specular) * irradiance * NdotL), 1.0); // TODO: miert kell a max
}

vec4 viewSpaceShading(vec3 position, vec3 albedo, vec3 N, float metallic, float roughness, vec3 lightDirection, vec3 irradiance)
{
    return anySpaceShading(normalize(-position), albedo, N, metallic, roughness, normalize(lightDirection), irradiance);
}

vec3 tonemapBasic(vec3 colour)
{
    return vec3(1.0) - exp(-colour * 0.9);
}

vec3 tonemapACES(vec3 colour)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((colour * (A * colour + B)) / (colour * (C * colour + D) + E), 0.0, 1.0);
}

vec3 tonemapUnchartedImpl(vec3 color)
{
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec3 tonemapUncharted(vec3 colour)
{
    const float W = 11.2;
    colour = tonemapUnchartedImpl(colour * 2.0);
    vec3 whiteScale = 1.0 / tonemapUnchartedImpl(vec3(W));
    return colour * whiteScale;
}
