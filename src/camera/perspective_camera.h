#ifndef VULKAN_INTRO_PERSPECTIVE_CAMERA_COMPONENT_H
#define VULKAN_INTRO_PERSPECTIVE_CAMERA_COMPONENT_H

class PerspectiveCamera
{
public:
	PerspectiveCamera();
    void updateGui();
    void update(double dt);

public:
	glm::mat4 V() const;
	glm::mat4 P(float aspectRatio) const;
	glm::mat4 VRH() const;
	glm::mat4 PRH(float aspectRatio) const;
	glm::vec3 eyePos() const { return pos; }

	glm::vec3 Up() const { return up; }
	glm::vec3 Right() const { return right; }
	float farPlane() const { return farClip; }
	float nearPlane() const { return nearClip; }
	float fov() const { return hFov; }

	std::array<glm::vec3, 8> furstumPoints(float aspectRatio) const;

	void moveTo(glm::vec3 pos);
	void lookAt(glm::vec3 target);

private:
	void updateVectors();

private:
	float hFov, nearClip, farClip;
	float yaw, pitch, roll;
	glm::vec3 front, up, right, worldUp, pos;

private:
	float movementSpeed = 0.5f, mouseSensitivity = 0.3f;
	bool directionChanging = false, firstFrameAfterDirChange = false;
	bool dirChanged = false;
	glm::vec2 lastCursorPos = { 0, 0 };
	float velocityFront = 0, velocityRight = 0, velocityUp = 0;
	float mouseDeltaVertical = 0, mouseDeltaHorizontal = 0;
};

#endif//VULKAN_INTRO_PERSPECTIVE_CAMERA_COMPONENT_H
