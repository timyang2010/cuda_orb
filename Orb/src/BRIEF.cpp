#pragma once
#include "BRIEF.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <intrin.h>
#include <bitset>
using namespace std;
using namespace cv;

namespace BRIEF
{

	BRIEF::BRIEF() : BRIEF(BRIEF_DEFAULT_WINDOW_SIZE)
	{

	}
	BRIEF::BRIEF(int S)
	{
		_tests = GenerateBinaryTests(BRIEF_DEFAULT_TEST_COUNT, S);
	}
	BRIEF::BRIEF(vector<BRIEF::BinaryTest>& ts)
	{
		_tests = ts;
	}
	vector<BRIEF::BinaryTest> BRIEF::GenerateBinaryTests(const int count, const int dim)
	{
		vector<BRIEF::BinaryTest> tests;
		int radius = dim / 2;
		for (int i = 0; i < count; ++i)
		{
			tests.push_back({ 
				int8_t(rand() % dim - radius),
				int8_t(rand() % dim - radius),
				int8_t(rand() % dim - radius),
				int8_t(rand() % dim - radius) 
			});
		}
		return tests;
	}

	std::vector<BRIEF::Feature> BRIEF::extractFeatures(uint8_t** image, vector<Point2d>& positions)  const
	{
		std::vector<BRIEF::Feature> features(positions.size());
		#pragma omp parallel for
		for (int p = 0; p < positions.size(); ++p)
		{
			Feature f;
			int bitpos = 0;
			vector<Point2d>::iterator it = positions.begin()+p;
			for (vector<BinaryTest>::const_iterator i = _tests.begin(); i != _tests.end(); ++i)
			{
				int x1 = it->x + i->x1; int y1 = it->y + i->y1;
				int x2 = it->x + i->x2; int y2 = it->y + i->y2;
				f.setbit(bitpos, image[y1][x1] > image[y2][x2]);
				++bitpos;
			}
			f.position = (*it);
			features[p] = f;
		}
		return features;
	}
	unsigned int BRIEF::DistanceBetween(Feature& f1, Feature& f2) const
	{
		return f1 - f2;
	}

	BRIEF::Feature::Feature()
	{
		memset(value, 0, sizeof(int) * BRIEF_DEFAULT_WORDLENGTH);
	}

	int BRIEF::Feature::operator- (Feature& feature) const
	{
		unsigned int sum_difeatures = 0;
		union {
			__m256 x;
			unsigned int y[BRIEF_DEFAULT_WORDLENGTH];
		};
		x = _mm256_xor_ps(f_vect, feature.f_vect);
		for (int i = 0; i < BRIEF_DEFAULT_WORDLENGTH; ++i)
		{
			sum_difeatures += __popcnt(y[i]);
		}
		return sum_difeatures;
	}
	void BRIEF::Feature::setbit(int pos, bool v)
	{
		value[(pos / 32)] |= v ? (1 << pos % 32) : 0;
	}

	BRIEF::BinaryTest BRIEF::BinaryTest::Rotate(double _cos,double _sin) const
	{
		double _x1 = (double)x1*_cos - (double)y1*_sin;
		double _y1 = (double)x1*_sin + (double)y1*_cos;
		double _x2 = (double)x2*_cos - (double)y2*_sin;
		double _y2 = (double)x2*_sin + (double)y2*_cos;
		return BinaryTest{(int8_t)_x1,(int8_t)_y1,(int8_t)_x2,(int8_t)_y2};
	}
	std::ostream& operator<<(std::ostream& os, const BRIEF::BinaryTest& dt)
	{
		os << (int)dt.x1 << " " << (int)dt.y1 << " " << (int)dt.x2 << " " << (int)dt.y2 << " ";
		return os;
	}
	
	ostream& operator<<(ostream& os, const BRIEF::Feature& f)
	{
		for (int i = 0; i < BRIEF_DEFAULT_WORDLENGTH; ++i)
		{
			os << bitset<32>(f.value[i]) << endl;
		}
		return os;
	}

}