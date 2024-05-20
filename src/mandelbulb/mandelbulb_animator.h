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
		return color0;
	}

	const glm::vec3& getColor1() const
	{
		return color1;
	}

	const glm::vec3& getColor2() const
	{
		return color2;
	}

	enum class AnimationWaveForm
	{
		off,
		animSine,
		animTriangle,
		animSaw,
		animPulse
	};

private:
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

	glm::vec3 skyColor = glm::vec3{ 0.1f, 0.15f, 0.3f };
	glm::vec3 horizontColor = glm::vec3{ 0.1f, 0.25f, 0.3f };
	glm::vec3 groundColor = glm::vec3{ 0.35f, 0.28f, 0.25f };
	glm::vec3 color0 = glm::vec3{ 0, 0, 1 };
	glm::vec3 color1 = glm::vec3{ 0.5, 1, 0.5 };
	glm::vec3 color2 = glm::vec3{ 1, 0, 0 };
};

inline MandelbulbAnimator& theMandelbulbAnimator = MandelbulbAnimator::instance();

#endif	// MANDELBULB_ANIMATOR_H