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

	const glm::vec3& getBackgroundColor() const
	{
		return backgroundColor;
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
	float minN = 2.0f;
	float maxN = 10.0f;
	float animationSpeedN = 0.5f;
	bool animationDirectionN = true;

	glm::vec3 backgroundColor = glm::vec3{ 0.1f, 0.1f, 0.2f };
};

inline MandelbulbAnimator& theMandelbulbAnimator = MandelbulbAnimator::instance();

#endif	// MANDELBULB_ANIMATOR_H