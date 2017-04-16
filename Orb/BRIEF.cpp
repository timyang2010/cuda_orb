#pragma once
#include "brief.h"
#include <opencv2\core.hpp>
#include <iostream>
#include <vector>
#include <intrin.h>

using namespace std;
using namespace cv;



BRIEF::BRIEF() : BRIEF(BRIEF_DEFAULT_WINDOW_SIZE)
{

}
BRIEF::BRIEF(int S)
{
	size = S;
	GenerateBinaryTests(xp, yp, 512, S);
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
BRIEF::Features BRIEF::extractFeature(unsigned char* image, vector<Point2d>& positions, const int width, const int height)  const
{
	int size = width*height;
	BRIEF::Features features = BRIEF::Features();
	for (vector<Point2d>::iterator it = positions.begin(); it != positions.end(); ++it)
	{
		Feature f;
#pragma omp parallel 
		for (int i = 0, bitpos = 0; i < 512; i += 2, ++bitpos)
		{
			int x1 = it->x + xp[i]; int y1 = it->y + yp[i];
			int x2 = it->x + xp[i + 1]; int y2 = it->y + yp[i + 1];
			int stp = y1*width + x1;
			int edp = y2*width + x2;

			if (stp>0 && stp<size && edp>0 && edp<size)
				f.setbit(bitpos, image[stp] > image[edp]);
		}
		f.position = (*it);
		features.push_back(f);
	}
	return features;
}
BRIEF::Features BRIEF::extractFeature(unsigned char** image, vector<Point2d>& positions, const int width, const int height)  const
{
	int size = width*height;
	BRIEF::Features features = BRIEF::Features();
	for (vector<Point2d>::iterator it = positions.begin(); it != positions.end(); ++it)
	{
		Feature f;
		for (int i = 0, bitpos = 0; i < 512; i += 2, ++bitpos)
		{
			int x1 = it->x + xp[i]; int y1 = it->y + yp[i];
			int x2 = it->x + xp[i + 1]; int y2 = it->y + yp[i + 1];
			
			f.setbit(bitpos, image[y1][x1] > image[y2][x2]);
		}
		f.position = (*it);
		features.push_back(f);
	}
	return features;
}
unsigned int BRIEF::DistanceBetween(Feature& f1, Feature& f2) const
{
	return f1 - f2;
}

BRIEF::Feature::Feature()
{
	for (int i = 0; i < 8; ++i)
	{
		value[i] = 0;
	}
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


#include <bitset>
ostream& operator<<(ostream& os, const BRIEF::Feature& f)
{
	for (int i = 0; i < 8; ++i)
	{
		os << bitset<32>(f.value[i]) << endl;
	}
	return os;
}