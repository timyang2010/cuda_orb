#include "BRIEF.h"
#include <math.h>
#include <iostream>
using namespace std;
namespace BRIEF
{

	Optimizer::Optimizer()
	{

	}
	void Optimizer::extractFeatures(uint8_t** image, std::vector<cv::Point2f>& positions)
	{
		
		#pragma omp parallel for
		for (int i = 0; i < candidates.size(); ++i)
		{
			candidate c = candidates[i];	
			for (std::vector<cv::Point2f>::iterator p = positions.begin(); p < positions.end(); ++p)
			{
				int x1 = p->x + c.test.x1; int y1 = p->y + c.test.y1;
				int x2 = p->x + c.test.x2; int y2 = p->y + c.test.y2;
				candidates[i].testResult.push_back(image[y1][x1] > image[y2][x2] ? 1 : 0);
			}
		}

	}
	void Optimizer::extractFeatures(uint8_t** image, std::vector<cv::Point2f>& positions, std::vector<float>& angles)
	{

	}

	//returns a set of optimized BRIEF tests based on input keypoints
	std::vector<BRIEF::BinaryTest> Optimizer::Optimize(int length)
	{
		#pragma omp parallel for
		for (int i = 0; i < candidates.size(); ++i)
		{
			candidates[i].computeRank();
		}
		std::sort(candidates.begin(), candidates.end(), [](candidate& c1, candidate& c2) -> bool {
			return c1.rank > c2.rank;
		});
		std::vector<BRIEF::BinaryTest> v;
		v.push_back(candidates[0].test);
		//greedy search algorithm

		return v;
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
		int padding = subWindowSize - 1;
		int bound = windowSize - padding;
		int radius = bound / 2;
		vector<BRIEF::BinaryTest> tests;
		vector<cv::Point2i> tps;
		for (int i = 0; i < bound; ++i)
		{
			for (int j = 0; j < bound; ++j)
			{
				tps.push_back(cv::Point2i(i - radius, j - radius));
			}
		}
		for (int i = 0; i < tps.size(); ++i)
		{
			for (int j = i+1; j < tps.size(); ++j)
			{
				candidates.push_back(candidate({ int8_t(tps[i].x),int8_t(tps[i].y), int8_t(tps[j].x), int8_t(tps[j].y) }));
			}
		}
		std::cout <<"generated " << candidates.size() <<" tests"<< endl;
	}
	double Optimizer::correlation(candidate& c1, candidate& c2)
	{
		double sxx, syy, sxy;
		double mean_x, mean_y;
		for (int i = 0; i < c1.testResult.size(); ++i)
		{
			mean_x += c1.testResult[i];
			mean_y += c2.testResult[i];
		}
		mean_x /= c1.testResult.size();
		mean_y /= c2.testResult.size();

		for (int i = 0; i < c1.testResult.size(); ++i)
		{
			sxx += pow(c1.testResult[i] - mean_x, 2);
			syy += pow(c2.testResult[i] - mean_y, 2);
			sxy += (c1.testResult[i] - mean_x)*(c2.testResult[i] - mean_y);
		}
		return sxy / (sxx*syy);
	}
	void Optimizer::candidate::computeRank()
	{
		double sum = 0;
		for (int i=0;i<testResult.size();++i)
		{
			sum += double(testResult[i]);
		}
		this->rank = abs(0.5 - sum / testResult.size());
	}

}