#include "BRIEF.h"
#include <math.h>
#include <iostream>
#include "Profiler.h"
using namespace std;
namespace ty
{

	Optimizer::Optimizer()
	{

	}
	void Optimizer::extractFeatures(uint8_t** image, std::vector<Keypoint>& positions)
	{	
	#pragma omp parallel for
		for (int i = 0; i < candidates.size(); ++i)
		{
			candidate c = candidates[i];
			for (int j = 0; j < positions.size(); ++j)
			{
				int ang = positions[j].z;
				int x1 = positions[j].x + c.tests[ang].x1; int y1 = positions[j].y + c.tests[ang].y1;
				int x2 = positions[j].x + c.tests[ang].x2; int y2 = positions[j].y + c.tests[ang].y2;
				candidates[i].testResult.push_back(image[y1][x1] > image[y2][x2] ? 1 : 0);
			}
		}
	}

	bool Optimizer::checkCorrelation(candidate& c1,vector<candidate>& v, double thres)
	{
		for (vector<candidate>::iterator jt = v.begin(); jt < v.end(); ++jt)
		{
			if (abs(correlation(c1, *jt)) > thres)
			{
				return false;
			}
		}
		return true;
	}

	//returns a set of optimized BRIEF tests based on input keypoints
	std::vector<BRIEF::BinaryTest> Optimizer::Optimize(float stp,float step)
	{
		#pragma omp parallel for
		for (int i = 0; i < candidates.size(); ++i)
		{
			candidates[i].computeRank();
		}
		std::sort(candidates.begin(), candidates.end(), [](candidate& c1, candidate& c2) -> bool {
			return c1.rank > c2.rank;
		});
		std::vector<candidate> v;
		vector<BRIEF::BinaryTest> tests;
		for (;v.size()<256;)
		{	
			v.clear();
			v.push_back(candidates[0]);
			int iters = 0;
			for (vector<candidate>::iterator it = candidates.begin() + 1; it < candidates.end(); ++it,++iters)
			{
				if (checkCorrelation(*it, v, stp))
				{
					v.push_back(*it);
				}
				if (v.size() > 255) break;
				cout << '\r'<<iters<<" "<<v.size()<< "/256    ";
			}
			cout << endl<< "achieved: " << v.size() << "/256  " << "threshold:" << stp << endl;
			stp += step;
		}
		cout << "training complete" << endl;
		for (int i = 0; i < 256; ++i)
		{
			tests.push_back(v[i].tests[0]);
		}
		return tests;
	}
	void Optimizer::generateTests(int windowSize, int subWindowSize,int min_distance,int scale)
	{
		int padding = subWindowSize - 1;
		int bound = windowSize - padding;
		int radius = bound / 2;
		vector<BRIEF::BinaryTest> tests;
		vector<cv::Point2i> tps;
		for (int i = -radius; i <= radius; ++i)
		{
			for (int j = -radius; j < radius; ++j)
			{
				tps.push_back(cv::Point2i(i*scale, j*scale));
			}
		}
		for (int i = 0; i < tps.size(); ++i)
		{
			for (int j = i+1; j < tps.size(); ++j)
			{
				tests.push_back({ int8_t(tps[i].x),int8_t(tps[i].y), int8_t(tps[j].x), int8_t(tps[j].y) });
			}
		}
		int dist = scale*min_distance;
		for (vector<BRIEF::BinaryTest>::iterator it = tests.begin(); it < tests.end(); ++it)
		{
			if (abs(it->x1 - it->x2) > dist || abs(it->y1 - it->y2) > dist)
			{
				candidates.push_back(candidate(*it));
			}
		}
	}
	double Optimizer::correlation(candidate& c1, candidate& c2)
	{
		double length = (double)c1.testResult.size();
		double mean_x = c1.mean(), mean_y=c2.mean();	
		double cov = 0;
		for (int i = 0; i < c1.testResult.size(); ++i)
		{
			cov += (c1.testResult[i] - mean_x)* (c2.testResult[i] - mean_y);
		}
		cov /= length;
		return cov/(c1.stddev()*c2.stddev());
	}
	double Optimizer::candidate::mean()
	{
		if (_mean == -1)
		{
			double m = 0;
			for (vector<unsigned short>::iterator it = testResult.begin(); it < testResult.end(); ++it)
			{
				m += (double)*it;
			}
			_mean = m / testResult.size();
		}
		return _mean;
	}
	double Optimizer::candidate::stddev()
	{
		if (_stddev == -1)
		{
			double u = mean();
			double s = 0;
			for (vector<unsigned short>::iterator it = testResult.begin(); it < testResult.end(); ++it)
			{
				s += pow((double)(*it) - u,2);
			}
			_stddev = sqrt(s/testResult.size());
		}
		return _stddev;
	}
	void Optimizer::candidate::computeRank()
	{
		double sum = 0;
		for (int i=0;i<testResult.size();++i)
		{
			sum += double(0.5 - testResult[i]);
		}
		this->rank = abs(sum / testResult.size());
	}

	Optimizer::candidate::candidate(BRIEF::BinaryTest _test)
	{
		double delta = 2 * 3.141596f / sBRIEF_DEFAULT_LUT_SIZE;
		double ang = 0;
		for (int i = 0; i <sBRIEF_DEFAULT_LUT_SIZE; ++i, ang += delta)
		{
			double _sin = sin(ang);
			double _cos = cos(ang);
			tests.push_back(_test.Rotate(_cos, _sin));
		}
		
	}

}