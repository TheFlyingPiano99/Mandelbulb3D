#ifndef MANDELBULB_ANIMATOR_H
#define MANDELBULB_ANIMATOR_H


class MandelbulbAnimator
{
public:

	static MandelbulbAnimator& instance()
	{
		static MandelbulbAnimator animator;
		return animator;
	}

	void updateGui();
	void update(double dt);

	float getN() const
	{
		return n;
	}

	const float getColoringMultiplier() const
	{
		return coloringMultiplier;
	}

	const float getColoringPower() const
	{
		return coloringPower;
	}

	const glm::vec3& getSkyColor() const
	{
		return skyColor;
	}

	const glm::vec3& getHorizontColor() const
	{
		return horizontColor;
	}

	const glm::vec3& getGroundColor() const
	{
		return groundColor;
	}

	const glm::vec3& getColor0() const
	{
		float t = getModulated(colorInterpolator0, modulationColor0, 0.0f, 1.0f);
		return color00 * (1.0f - t) + color10 * t;
	}

	const glm::vec3& getColor1() const
	{
		float t = getModulated(colorInterpolator1, modulationColor1, 0.0f, 1.0f);
		return color01 * (1.0f - t) + color11 * t;
	}

	const glm::vec3& getColor2() const
	{
		float t = getModulated(colorInterpolator2, modulationColor2, 0.0f, 1.0f);
		return color02 * (1.0f - t) + color12 * t;
	}

	float getRotation() const
	{
		return rotation;
	}

	enum class AnimationWaveForm
	{
		off,
		animSine,
		animTriangle,
		animSaw,
		animPulse
	};

	enum class ModulationType
	{
		modOff,
		modFrequency,
		modAmplitude
	};

private:
	float getModulated(float baseVal, float modAmp, float min = 0.0f, float max = 1.0f) const;

	float n = 8.0f;
	AnimationWaveForm animationWaveFormN = AnimationWaveForm::animTriangle;
	float minN = 4.0f;
	float maxN = 20.0f;
	float animationSpeedN = 0.5f;
	bool animationDirectionN = true;
	float modulationN = 0.0f;
	float coloringMultiplier = 1.0f;
	float coloringPower = 0.9f;

	float modulationFrequency = 1.0f;
	double modulationT = 0.0;

	ModulationType secondaryModulationType = ModulationType::modFrequency;
	float secondaryModulationFrequency = 1.0f;
	float secondaryModulationAmplitude = 0.0f;
	double secondaryModulationT = 0.0;

	glm::vec3 skyColor = glm::vec3{ 0.1f, 0.15f, 0.3f };
	glm::vec3 horizontColor = glm::vec3{ 0.1f, 0.25f, 0.3f };
	glm::vec3 groundColor = glm::vec3{ 0.35f, 0.28f, 0.25f };
	
	glm::vec3 color00 = glm::vec3{ 0, 0, 1 };
	glm::vec3 color01 = glm::vec3{ 0.5, 1, 0.5 };
	glm::vec3 color02 = glm::vec3{ 1, 0, 0 };

	glm::vec3 color10 = glm::vec3{ 0, 0, 1 };
	glm::vec3 color11 = glm::vec3{ 0.5, 1, 0.5 };
	glm::vec3 color12 = glm::vec3{ 1, 0, 0 };

	float colorInterpolator0 = 0.0f;
	float colorInterpolator1 = 0.0f;
	float colorInterpolator2 = 0.0f;

	float modulationColor0 = 0.0f;
	float modulationColor1 = 0.0f;
	float modulationColor2 = 0.0f;

	float rotation = 0.0f;
	float rotationSpeed = 0.0f;
	float rotationModulationAmp = 0.0f;
};

inline MandelbulbAnimator& theMandelbulbAnimator = MandelbulbAnimator::instance();

#endif	// MANDELBULB_ANIMATOR_H