#include "mandelbulb_animator.h"
#include <iostream>

# define M_PI           3.14159265358979323846


void MandelbulbAnimator::update(double dt)
{
	if (isHighFidelityHold)
		return;

	switch (animationWaveFormN) {
	case AnimationWaveForm::off:
		break;
	case AnimationWaveForm::animTriangle:
	{
		n += ((animationDirectionN) ? 1.0f : -1.0f) * getModulated(animationSpeedN, modulationN, 0.0f, 20.0f) * dt;
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

	rotation += getModulated(rotationSpeed, rotationModulationAmp, -6.0f, 6.0f) * dt;

	// Update LFO-s:
	modulationT += dt;
	secondaryModulationT += dt;
}

float MandelbulbAnimator::getModulated(float baseVal, float modAmp, float min, float max) const {
	float x;
	float secondaryX = secondaryModulationT * secondaryModulationFrequency * 2.0 * M_PI;
	if (secondaryModulationType == ModulationType::modFrequency) {
		x = modulationT * (modulationFrequency + secondaryModulationAmplitude * sinf(secondaryX)) * 2.0 * M_PI;
	}
	else {
		x = modulationT * modulationFrequency * 2.0 * M_PI;
	}
	float amplitude = modAmp + ((secondaryModulationType == ModulationType::modAmplitude) ? secondaryModulationAmplitude * sinf(secondaryX) : 0.0f);
	return fminf(fmaxf(baseVal + amplitude * sinf(x), min), max);
}

void MandelbulbAnimator::updateGui()
{
	ImGui::Text("Shading");
	ImGui::ColorEdit3("Dir. light power", (float*)&lightPower);
	ImGui::ColorEdit3("Ambient power", (float*)&ambientPower);
	ImGui::DragFloat("Edge intensity", &edgeIntensity, 0.01f, 0.0f, 1.0f, "%.2f");
	ImGui::DragFloat("Diffuse intensity", &diffuseIntensity, 0.01f, 0.0f, 10.0f, "%.2f");
	ImGui::DragFloat("Specular intensity", &specularIntensity, 0.01f, 0.0f, 10.0f, "%.2f");
	ImGui::DragFloat("Ambient intensity", &ambientIntensity, 0.01f, 0.0f, 10.0f, "%.2f");
	ImGui::DragFloat("Shininess", &shininess, 1.0f, 1.0f, 100.0f, "%.0f");
	ImGui::DragFloat("Opacity scale", &opacityScale, 0.1f, 1.0f, 100.0f, "%.1f");

	ImGui::Text("Bulb movement");
	ImGui::DragFloat("Bulb rotation speed", &rotationSpeed, 0.1f, -6.0f, 6.0f, "%.1f");

	ImGui::DragFloat("n animation speed", &animationSpeedN, 0.1f, 0.0f, 20.0f, "%.1f");
	ImGui::DragFloat("min n", &minN, 0.1f, 1.0f, maxN, "%.1f");
	ImGui::DragFloat("max n", &maxN, 0.1f, minN, 100.0f, "%.1f");

	ImGui::ColorEdit3("Sky color", (float*)&skyColor);
	ImGui::ColorEdit3("Horizont color", (float*)&horizontColor);
	ImGui::ColorEdit3("Ground color", (float*)&groundColor);

	ImGui::Text("Bulb color A");
	ImGui::ColorEdit3("Bulb color A [0.0]", (float*)&color00);
	ImGui::ColorEdit3("Bulb color A [0.5]", (float*)&color01);
	ImGui::ColorEdit3("Bulb color A [1.0]", (float*)&color02);

	ImGui::Text("Bulb color B");
	ImGui::ColorEdit3("Bulb color B [0.0]", (float*)&color10);
	ImGui::ColorEdit3("Bulb color B [0.5]", (float*)&color11);
	ImGui::ColorEdit3("Bulb color B [1.0]", (float*)&color12);

	ImGui::Text("Bulb color mix A <-> B");
	ImGui::DragFloat("Color mix A <-> B [0.0]", &colorInterpolator0, 0.1f, 0.0f, 1.0f, "%.1f");
	ImGui::DragFloat("Color mix A <-> B [0.5]", &colorInterpolator1, 0.1f, 0.0f, 1.0f, "%.1f");
	ImGui::DragFloat("Color mix A <-> B [1.0]", &colorInterpolator2, 0.1f, 0.0f, 1.0f, "%.1f");

	ImGui::DragFloat("Coloring multiplier", &coloringMultiplier, 0.1f, 0.1f, 10.0f, "%.1f");
	ImGui::DragFloat("Coloring power", &coloringPower, 0.1f, 0.1f, 1.0f, "%.1f");

	ImGui::Text("Modulation control");
	ImGui::DragFloat("Modulation frequency", &modulationFrequency, 0.1f, 0.1f, 10.0f, "%.1f");
	ImGui::DragFloat("Secondary modulation frequency", &secondaryModulationFrequency, 0.001f, 0.001f, 1.0f, "%.3f");
	ImGui::DragFloat("Secondary modulation amplitude", &secondaryModulationAmplitude, 0.001f, 0.0f, 1.0f, "%.3f");

	ImGui::Text("Modulation routing");
	ImGui::DragFloat("Modulation for rotation speed", &rotationModulationAmp, 0.1f, 0.0f, 10.0f, "%.1f");
	ImGui::DragFloat("Modulation for n anim. speed", &modulationN, 0.1f, 0.0f, 20.0f, "%.1f");
	ImGui::DragFloat("Modulation color mix [0.0]", &modulationColor0, 0.1f, 0.0f, 1.0f, "%.1f");
	ImGui::DragFloat("Modulation color mix [0.5]", &modulationColor1, 0.1f, 0.0f, 1.0f, "%.1f");
	ImGui::DragFloat("Modulation color mix [1.0]", &modulationColor2, 0.1f, 0.0f, 1.0f, "%.1f");

	bool isPressed = ImGui::Button("High fidelity render");
	if (isPressed && !isHighFidelityHold) {
		isHighFidelityRender = true;
	}
	isPressed = ImGui::Button("Resume");
	if (isPressed && isHighFidelityHold) {
		isHighFidelityHold = false;
	}

}
