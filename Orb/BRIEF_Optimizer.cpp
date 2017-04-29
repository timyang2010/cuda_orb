#include "BRIEF.h"
#include <math.h>
using namespace std;
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
	double Optimizer::computeVariance(std::vector<BRIEF::Feature>& features)
	{
		double mean = 0,var = 0;
		for (vector<BRIEF::Feature>::iterator f = features.begin(); f < features.end(); ++f)
		{
			for(int i=0;i<8;++i)
				mean += __popcnt(f->value[i]);
		}
		mean /= features.size();
		for (vector<BRIEF::Feature>::iterator f = features.begin(); f < features.end(); ++f)
		{
			int sum = 0;
			for (int i = 0; i<8; ++i)
				sum += __popcnt(f->value[i]);
			var += pow(sum - mean,2);
		}
		return var / features.size();
	}
	void Optimizer::generateTests(int windowSize, int subWindowSize)
	{

	}
	double Optimizer::correlation(candidate& c1, candidate& c2)
	{

	}
	void Optimizer::sort(double mean)
	{

	}

}