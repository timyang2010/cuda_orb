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
		size = S;
		_tests = GenerateBinaryTests(BRIEF_DEFAULT_TEST_COUNT, S);
		//GenerateBinaryTests(xp, yp, 512, S);
	}
	void BRIEF::GenerateBinaryTests(int* x, int* y, const int count, const int dim)
	{
		int radius = dim / 2;
		for (int i = 0; i < count; ++i)
		{
			x[i] = rand() % dim - radius;
			y[i] = rand() % dim - radius;
		}
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
	BRIEF::Features BRIEF::extractFeature(uint8_t* image, vector<Point2d>& positions, const int width, const int height)  const
	{
		int size = width*height;
		BRIEF::Features features;
		#pragma omp parallel for
		for (int p = 0; p < positions.size(); ++p)
		{
			Feature f;
			int bitpos = 0;
			vector<Point2d>::iterator it = positions.begin() + p;
			for (vector<BinaryTest>::const_iterator i = _tests.begin(); i != _tests.end(); ++i)
			{
				int x1 = it->x + i->x1; int y1 = it->y + i->y1;
				int x2 = it->x + i->x2; int y2 = it->y + i->y2;
				int stp = y1*width + x1;
				int edp = y2*width + x2;
				if (stp > 0 && stp < size && edp>0 && edp < size)
					f.setbit(bitpos, image[stp] > image[edp]);
				++bitpos;
			}
			f.position = (*it);
			features[p] = f;
		}
		return features;
	}
	BRIEF::Features BRIEF::extractFeature(uint8_t** image, vector<Point2d>& positions, const int width, const int height)  const
	{
		int size = width*height;
		BRIEF::Features features(positions.size());
		#pragma omp parallel for
		for (int p = 0; p < positions.size(); ++p)
		{
			Feature f;
			int bitpos = 0;
			vector<Point2d>::iterator it = positions.begin()+p;
			for (vector<BinaryTest>::const_iterator i = _tests.begin(); i != _tests.end(); ++i)
			{
				if (it->x > 16 && it->x < width - 16 && it->y>16 && it->y < height - 16)
				{
					int x1 = it->x + i->x1; int y1 = it->y + i->y1;
					int x2 = it->x + i->x2; int y2 = it->y + i->y2;
					f.setbit(bitpos, image[y1][x1] > image[y2][x2]);
				}
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
		memset(value, 0, sizeof(int) * 8);
	}

	int BRIEF::Feature::operator- (Feature& feature) const
	{
		unsigned int sum_difeatures = 0;
		union {
			__m256 x;
			unsigned int y[8];
		};
		x = _mm256_xor_ps(f_vect, feature.f_vect);
		for (int i = 0; i < 8; ++i)
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
		double _x1 = x1*_cos - y1*_sin;
		double _y1 = x1*_sin + y1*_cos;
		double _x2 = x2*_cos - y2*_sin;
		double _y2 = x2*_sin + y2*_cos;
		return BinaryTest{(int8_t)_x1,(int8_t)_y1,(int8_t)_x2,(int8_t)_y2};
	}

	
	ostream& operator<<(ostream& os, const BRIEF::Feature& f)
	{
		for (int i = 0; i < 8; ++i)
		{
			os << bitset<32>(f.value[i]) << endl;
		}
		return os;
	}

}