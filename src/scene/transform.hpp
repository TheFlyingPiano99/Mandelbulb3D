#ifndef VULKAN_INTRO_TRANSFORM_HPP
#define VULKAN_INTRO_TRANSFORM_HPP

class Transform {
public:
    Transform() {
        set(glm::mat4(1));
    }

    const glm::quat& rotation() const { return mRotation; }
    const glm::vec3& position() const { return mPosition; }
    const glm::vec3& scale() const { return mScale; }

    void setRotation(glm::quat p) { mRotation = p; updateMatrix(); }
    void setPosition(glm::vec3 p) { mPosition = p; updateMatrix(); }
    void setScale(glm::vec3 p) { mScale = p; updateMatrix(); }
    void set(glm::vec3 position, glm::quat rotation, glm::vec3 scale) {
        mPosition = position; mRotation = rotation; mScale = scale;
        updateMatrix();
    }
    void set(glm::mat4 transform) {
        glm::vec3 scale;
        glm::quat rotation;
        glm::vec3 translation;
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(transform, scale, rotation, translation, skew, perspective);
        set(translation, rotation, scale);
    }

    const glm::mat4& M() const { return modelTr; }
    const glm::mat4& MInv() const { return modelTrInv; }

    friend class GameScene;

private:
    void updateMatrix() {
        modelTr = glm::mat4{ 1.0f };
        modelTr = glm::translate(modelTr, mPosition);
        modelTr = modelTr * glm::toMat4(mRotation);
        modelTr = glm::scale(modelTr, mScale);
        modelTrInv = glm::inverse(modelTr);
    }
    friend class GameObject;

private:
    glm::quat mRotation;
    glm::vec3 mPosition, mScale;
    glm::mat4 modelTr, modelTrInv;
};

#endif//VULKAN_INTRO_TRANSFORM_HPP
