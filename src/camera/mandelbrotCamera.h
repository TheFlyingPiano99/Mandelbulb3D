#ifndef VULKAN_INTRO_MANDELBROT_CAMERA_COMPONENT_H
#define VULKAN_INTRO_MANDELBROT_CAMERA_COMPONENT_H

class MandelbrotCamera {
public:
	void initialize();

	void update(double dt);

	double x, y = 0.0;
	double zoom = 1.0;

private:
	glm::vec2 lastCursorPos = {0.0f, 0.0f};
	double mouseSensitivity = 0.05;
	double mouseZoomSensitivity = 0.1;
	double mouseDeltaVertical = 0.0;
	double mouseDeltaHorizontal = 0.0;
	bool isGrabbed = false;
};

inline float MandelbrotX, MandelbrotY = 0.0f;
inline float MandelbrotZoom = 1.0f;

#endif VULKAN_INTRO_MANDELBROT_CAMERA_COMPONENT_H