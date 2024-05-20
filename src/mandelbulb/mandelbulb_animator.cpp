#include "mandelbulb_animator.h"
#include <iostream>

void MandelbulbAnimator::update(double dt)
{
	switch (animationWaveFormN) {
	case AnimationWaveForm::off:
		break;
	case AnimationWaveForm::animTriangle:
	{
		n += ((animationDirectionN) ? 1.0f : -1.0f) * (animationSpeedN + fmaxf(modulationN * sin(modulationT * modulationFrequency * 2.0 * M_PI), 0.0f))  * dt;
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
		// TODO
		break;
	}
	}
	modulationT += dt;
}

void MandelbulbAnimator::updateGui()
{
	ImGui::DragFloat("min n", &minN, 0.1f, 1.0f, maxN, "%.1f");
	ImGui::DragFloat("max n", &maxN, 0.1f, minN, 100.0f, "%.1f");
	ImGui::DragFloat("n animation speed", &animationSpeedN, 0.1f, 0.0f, 10.0f, "%.1f");

	ImGui::DragFloat("Coloring multiplier", &coloringMultiplier, 0.1f, 0.1f, 4.0f, "%.1f");
	ImGui::DragFloat("Coloring power", &coloringPower, 0.1f, 0.1f, 1.0f, "%.1f");

	ImGui::ColorEdit3("Sky color", (float*)&skyColor);
	ImGui::ColorEdit3("Horizont color", (float*)&horizontColor);
	ImGui::ColorEdit3("Ground color", (float*)&groundColor);

	ImGui::ColorEdit3("Bulb color 0", (float*)&color0);
	ImGui::ColorEdit3("Bulb color 1", (float*)&color1);
	ImGui::ColorEdit3("Bulb color 2", (float*)&color2);

	ImGui::DragFloat("Modulation frequency", &modulationFrequency, 0.1f, 0.1f, 10.0f);
	ImGui::DragFloat("Modulation ampltude for n", &modulationN, 0.1f, 0.1f, 10.0f);
}
