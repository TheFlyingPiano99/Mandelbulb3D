#include "perspective_camera.h"
#include "gui/gui_manager.h"

using namespace glfwim;

PerspectiveCamera::PerspectiveCamera()
{
	worldUp = { 0.0, 1.0, 0.0 };

	pos = glm::vec3{ 0, 0, 0 };
	nearClip = 0.1f;
	farClip = 1000.f;
	hFov = 45.f;
	yaw = 0.f;
	pitch = 0.f;
	roll = 0.f;

	updateVectors();

	theInputManager.registerUtf8KeyHandler("w", [&](auto /* mod */, auto action) {
		if (action == Action::Press) {
			velocityFront -= 1.0f;
		} else if (action == Action::Release) {
			velocityFront += 1.0f;
		}
	});

	theInputManager.registerUtf8KeyHandler("s", [&](auto /* mod */, auto action) {
		if (action == Action::Press) {
			velocityFront += 1.0f;
		} else if (action == Action::Release) {
			velocityFront -= 1.0f;
		}
	});

	theInputManager.registerUtf8KeyHandler("a", [&](auto /* mod */, auto action) {
		if (action == Action::Press) {
			velocityRight += 1.0f;
		} else if (action == Action::Release) {
			velocityRight -= 1.0f;
		}
	});

	theInputManager.registerUtf8KeyHandler("d", [&](auto /* mod */, auto action) {
		if (action == Action::Press) {
			velocityRight -= 1.0f;
		} else if (action == Action::Release) {
			velocityRight += 1.0f;
		}
	});

	theInputManager.registerMouseButtonHandler(MouseButton::Left, [&](auto /* mod */, auto action) {
		if (action == Action::Press) {
			if (!theGUIManager.isMouseCaptured()) {
				if (!theGUIManager.forceCursor()) {
					theInputManager.setMouseMode(MouseMode::Disabled);
				}
				directionChanging = true;
				firstFrameAfterDirChange = true;
				ImGuizmo::Enable(false);
			}
		} else if (action == Action::Release) {
			theInputManager.setMouseMode(MouseMode::Enabled);
			directionChanging = false;
			ImGuizmo::Enable(true);
		}
	});

	theInputManager.registerCursorPositionHandler([&](double x, double y) {
		if (directionChanging && !firstFrameAfterDirChange) {
			auto delta = glm::vec2{ (float)x, (float)y } - lastCursorPos;
			mouseDeltaVertical = -delta.y * mouseSensitivity;
			mouseDeltaHorizontal = delta.x * mouseSensitivity;
			dirChanged = true;
		} else {
			mouseDeltaVertical = mouseDeltaHorizontal = 0;
		}
		firstFrameAfterDirChange = false;
		lastCursorPos = { x, y };
	});

	theInputManager.registerMouseScrollHandler([&](double /* x */, double y) {
		movementSpeed += (float)y / 10.f;
		movementSpeed = glm::clamp(movementSpeed, 0.1f, 50.f);
	});

	theGUIManager.registerMouseCaptureStateChangedHandler([&](bool isMouseCaptured) {
		if (isMouseCaptured) {
			theInputManager.setMouseMode(MouseMode::Enabled);
			directionChanging = false;
			ImGuizmo::Enable(true);
		}
	});
}

void PerspectiveCamera::update(double dt)
{
	if (!dirChanged) mouseDeltaVertical = mouseDeltaHorizontal = 0;
	dirChanged = false;

	bool isTurning = true;//!glm::all(glm::lessThan(glm::abs(glm::vec2{cih.mouseDeltaVertical, cih.mouseDeltaVertical}), glm::vec2{0.0001}));
	if (isTurning) {
		pitch += mouseDeltaVertical;
		yaw += mouseDeltaHorizontal;
		pitch = std::min(std::max(pitch, -89.f), 89.f);
		updateVectors();
	}

	auto velocity = glm::vec3{ velocityRight, velocityUp, velocityFront };
	bool isMoving = !glm::all(glm::lessThan(glm::abs(velocity), glm::vec3{ 0.1f }));
	if (isMoving) {
		moveTo(pos - (velocityRight * right + velocityUp * up + velocityFront * front) * (float)dt * movementSpeed);
	}
}

void PerspectiveCamera::updateGui()
{
	bool mod = false;
	mod |= ImGui::DragFloat("yaw", &yaw, 0.1f, -180, 180);
	mod |= ImGui::DragFloat("pitch", &pitch, 0.1f, -89, 89);
	mod |= ImGui::DragFloat("near clip", &nearClip, 0.01f, 0.01f, 1000);
	mod |= ImGui::DragFloat("far clip", &farClip, 0.01f, 0.01f, 1000);
	if (mod) {
		updateVectors();
	}
}

glm::mat4 PerspectiveCamera::V() const
{
	return glm::lookAt(pos, pos + front, up);
}

glm::mat4 PerspectiveCamera::VRH() const
{
	return glm::lookAtRH(pos, pos + front, up);
}

glm::mat4 PerspectiveCamera::P(float aspectRatio) const
{
	auto p = glm::perspective(glm::radians(hFov), aspectRatio, nearClip, farClip);
	p[1][1] *= -1; // Vulkan workaround
	return p;
}

glm::mat4 PerspectiveCamera::PRH(float aspectRatio) const
{
	auto p = glm::perspectiveRH(glm::radians(hFov), aspectRatio, nearClip, farClip);
	return p;
}

std::array<glm::vec3, 8> PerspectiveCamera::furstumPoints(float aspectRatio) const
{
	std::array<glm::vec3, 8> ret;

	float mRatio = aspectRatio;

	float nearWidth = 2 * tan(glm::radians(hFov) / 2) * nearClip;
	float nearHeight = nearWidth / mRatio;

	float farWidth = 2 * tan(glm::radians(hFov) / 2) * farClip;
	float farHeight = farWidth / mRatio;

	glm::vec3 fc = pos + front * farClip;
	glm::vec3 nc = pos + front * nearClip;

	ret[0] = fc + (up * farHeight / 2.0f) - (right * farWidth / 2.0f);
	ret[1] = fc + (up * farHeight / 2.0f) + (right * farWidth / 2.0f);
	ret[2] = fc - (up * farHeight / 2.0f) - (right * farWidth / 2.0f);
	ret[3] = fc - (up * farHeight / 2.0f) + (right * farWidth / 2.0f);

	ret[4] = nc + (up * nearHeight / 2.0f) - (right * nearWidth / 2.0f);
	ret[5] = nc + (up * nearHeight / 2.0f) + (right * nearWidth / 2.0f);
	ret[6] = nc - (up * nearHeight / 2.0f) - (right * nearWidth / 2.0f);
	ret[7] = nc - (up * nearHeight / 2.0f) + (right * nearWidth / 2.0f);

	return ret;
}

void PerspectiveCamera::updateVectors()
{
	glm::vec3 f;
	f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	f.y = sin(glm::radians(pitch));
	f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

	front = glm::normalize(f);
	right = glm::normalize(glm::cross(front, worldUp));
	up = glm::normalize(glm::cross(right, front));
}

void PerspectiveCamera::lookAt(glm::vec3 target)
{
	using namespace std::numbers;
	auto dir = glm::normalize(target - pos);
	auto pitch_rad = asin(dir.y);
	auto yaw_rad = asin(dir.z / cos(pitch_rad));
	if (dir.x < 0) {
		if (yaw_rad < 0) yaw_rad += 2 * (-(float)pi / 2.0f - yaw_rad);
		if (yaw_rad >= 0) yaw_rad += 2 * ((float)pi / 2.0f - yaw_rad);
	}
	auto oYaw = yaw, oPitch = pitch;

	yaw = glm::degrees(yaw_rad);
	pitch = glm::degrees(pitch_rad);

	if (std::isnan(yaw)) yaw = oYaw;
	if (std::isnan(pitch)) pitch = oPitch;

	updateVectors();
}

void PerspectiveCamera::moveTo(glm::vec3 target)
{
	pos = target;
}
