#include "BRIEF.h"
namespace BRIEF
{

	Optimizer::Optimizer()
	{

	}
	void Optimizer::extractFeatures(uint8_t** image, std::vector<cv::Point2d>& positions)
	{

	}
	void Optimizer::extractFeatures(uint8_t** image, std::vector<cv::Point2d>& positions, std::vector<float>& angles)
	{

	}

	//returns a set of optimized BRIEF tests based on input keypoints
	std::vector<BRIEF::BinaryTest> Optimizer::Optimize(int length)
	{

	}
	double Optimizer::variance(std::vector<BRIEF::Feature>& features)
	{

	}
	void Optimizer::generateTests(int windowSize = BRIEF_DEFAULT_WINDOW_SIZE, int subWindowSize = 5)
	{

	}
	double Optimizer::correlation(candidate& c1, candidate& c2)
	{

	}
	void Optimizer::sort(double mean)
	{

	}

}