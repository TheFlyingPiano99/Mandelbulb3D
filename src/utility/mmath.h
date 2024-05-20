#ifndef VULKAN_INTRO_M_MATH_H
#define VULKAN_INTRO_M_MATH_H

struct AABB {
    glm::vec3 bl, tr;

    static AABB max();
    static AABB min();
    AABB operator*(const glm::mat4 m) const;
    AABB merge(const AABB& aabb) const;
    glm::vec3 center() const;
    std::array<glm::vec3, 8> points() const;
};

#endif//VULKAN_INTRO_M_MATH_H
