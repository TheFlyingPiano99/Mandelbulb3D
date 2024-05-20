#include "mandelbrotCamera.h"
#include "../glfwim/input_manager.h"
#include "../glfwim/input_manager.hpp"
#include <iostream>
#include "gui/gui_manager.h"


void MandelbrotCamera::initialize()
{
	theInputManager.registerCursorPositionHandler([&](double x, double y) {
		auto delta = glm::vec2{ x, y } - lastCursorPos;
		mouseDeltaVertical = -delta.y * mouseSensitivity;
		mouseDeltaHorizontal = -delta.x * mouseSensitivity;
		lastCursorPos = { x, y };
	});

	theInputManager.registerMouseButtonHandler(glfwim::MouseButton::Left, [&](auto /* mod */, auto action) {
		if (action == glfwim::Action::Press) {
			if (!theGUIManager.isMouseCaptured()) {
				isGrabbed = true;
			}
		}
		else if (action == glfwim::Action::Release) {
			isGrabbed = false;
		}
	});

	theInputManager.registerMouseScrollHandler([&](double /* x */, double y) {
		zoom += y * mouseZoomSensitivity;
		if (zoom < 1.0) {
			zoom = 1.0;
		}
	});

}

void MandelbrotCamera::update(double dt)
{
	if (isGrabbed) {
		y += mouseDeltaVertical;
		x += mouseDeltaHorizontal;
		mouseDeltaVertical = 0.0f;
		mouseDeltaHorizontal = 0.0f;
		MandelbrotX = x;
		MandelbrotY = y;
		std::cout << x << ", " << y << std::endl;
	}
	MandelbrotZoom = zoom;
}
