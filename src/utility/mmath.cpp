#include "mmath.h"

AABB AABB::max()
{
    auto fmin = std::numeric_limits<float>::max() * -1.f;
    auto fmax = std::numeric_limits<float>::max();
    return AABB{glm::vec3{fmin, fmin, fmin},glm::vec3{fmax,fmax,fmax}};
}

AABB AABB::min()
{
    auto fmin = std::numeric_limits<float>::max() * -1.f;
    auto fmax = std::numeric_limits<float>::max();
    return AABB{glm::vec3{fmax,fmax,fmax},glm::vec3{fmin,fmin,fmin}};
}

AABB AABB::operator*(const glm::mat4 m) const
{
    std::array<glm::vec4, 8> points = {
        glm::vec4{bl.x, bl.y, bl.z, 1.0},
        glm::vec4{bl.x, bl.y, tr.z, 1.0},
        glm::vec4{bl.x, tr.y, bl.z, 1.0},
        glm::vec4{bl.x, tr.y, tr.z, 1.0},
        glm::vec4{tr.x, bl.y, bl.z, 1.0},
        glm::vec4{tr.x, bl.y, tr.z, 1.0},
        glm::vec4{tr.x, tr.y, bl.z, 1.0},
        glm::vec4{tr.x, tr.y, tr.z, 1.0}
    };

    for (auto& it : points) it = m * it;

    AABB ret;
    ret.bl = ret.tr = glm::vec3{points[0].x, points[0].y, points[0].z};
    for (auto& it : points) {
        ret.bl = glm::min(ret.bl, glm::vec3{it.x, it.y, it.z});
        ret.tr = glm::max(ret.tr, glm::vec3{it.x, it.y, it.z});
    }
    return ret;
}

AABB AABB::merge(const AABB& aabb) const
{
    AABB ret = *this;
    ret.bl = glm::min(ret.bl, aabb.bl);
    ret.tr = glm::max(ret.tr, aabb.tr);
    return ret;
}

glm::vec3 AABB::center() const
{
    return (bl + tr) / 2.f;
}

std::array<glm::vec3, 8> AABB::points() const
{
    return std::array<glm::vec3, 8> {
        glm::vec3{bl.x, bl.y, bl.z},
        glm::vec3{bl.x, bl.y, tr.z},
        glm::vec3{bl.x, tr.y, bl.z},
        glm::vec3{bl.x, tr.y, tr.z},
        glm::vec3{tr.x, bl.y, bl.z},
        glm::vec3{tr.x, bl.y, tr.z},
        glm::vec3{tr.x, tr.y, bl.z},
        glm::vec3{tr.x, tr.y, tr.z}
    };
}
