#include "mandelbulb_animator.h"
#include <iostream>

void MandelbulbAnimator::update(double dt)
{
	switch (animationWaveFormN) {
	case AnimationWaveForm::off:
		break;
	case AnimationWaveForm::animTriangle:
	{
		n += ((animationDirectionN) ? 1.0f : -1.0f) * animationSpeedN * dt;
		if (animationDirectionN)
		{
			if (n >= maxN) {
				n = maxN;
				animationDirectionN = false;
			}
		}
		else {
			if (n <= minN) {
				n = minN;
				animationDirectionN = true;
			}
		}
		break;
	}
	case AnimationWaveForm::animSine: {
		n += cos(animationSpeedN * dt * (n - minN) / (maxN - minN)) * animationSpeedN * dt;
		break;
	}
	}
}

void MandelbulbAnimator::updateGui()
{
	ImGui::DragFloat("min n", &minN, 0.1f, 1.0f, maxN, "%.1f");
	ImGui::DragFloat("max n", &maxN, 0.1f, minN, 100.0f, "%.1f");
	ImGui::DragFloat("n animation speed", &animationSpeedN, 0.1f, 0.0f, 10.0f, "%.1f");
	//std::cout << "Updated GUI" << std::endl;
}
