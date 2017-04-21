#pragma once
#include "brief.h"
#include <opencv2\core.hpp>
#include <iostream>
#include <vector>
#include <immintrin.h>
#define M_PI 3.14159265358979323846
using namespace std;
using namespace cv;

rBRIEF::rBRIEF() : rBRIEF(BRIEF_DEFAULT_WINDOW_SIZE*2)
{

}
rBRIEF::rBRIEF(int S) : rBRIEF(S, BRIEF_DEFAULT_TEST_COUNT*2)
{

}
rBRIEF::rBRIEF(int S,int count)
{
	size = S;
	int xp[512],yp[512];
	BRIEF::GenerateBinaryTests(xp, yp, count, size);
	generateLUT(512, S, sBRIEF_DEFAULT_LUT_SIZE,xp,yp);
}
rBRIEF::rBRIEF(int S, int* _xp, int* _yp)
{
	size = S;
	generateLUT(512, S, sBRIEF_DEFAULT_LUT_SIZE, _xp, _yp);
}
rBRIEF::~rBRIEF()
{

}


void rBRIEF::generateLUT(const int count, const int dim, const int num_angle,int* xp,int* yp)
{
	angleCount = num_angle;
	
	double delta = 2 * M_PI / num_angle;
	double ang = 0;
	for (int i = 0; i < num_angle; ++i, ang += delta)
	{
		double _sin = sin(ang);
		double _cos = cos(ang);
		vector<int> x;
		vector<int> y;

		for (int j = 0; j < count; ++j)
		{
			x.push_back((double)xp[j] * _cos - (double)yp[j] * _sin);
			y.push_back((double)(xp[j]) * _sin + (double)yp[j] * _cos);
		}
		lutx.push_back(x);
		luty.push_back(y);
	}
}
pair<vector<int>, vector<int>> rBRIEF::operator [](int i) const
{
	return pair<vector<int>, vector<int>>(lutx[i], luty[i]);
}
BRIEF::Features rBRIEF::extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, const int width, const int height) const
{
	Features features;
	int size = width*height;
	vector<int> xp = lutx[0];
	vector<int> yp = luty[0];
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

BRIEF::Features rBRIEF::extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, vector<float>& angles, const int width, const int height) const
{
	Features features = Features(positions.size());
	int size = width*height;
	float delta = 2 * M_PI / angleCount;
	//assert(positions.size() == angles.size());
	int length = positions.size();
#pragma omp parallel for
	for (int i = 0; i < length; ++i)
	{
		int ang = angles[i];
		vector<int> xp = lutx[ang];
		vector<int> yp = luty[ang];
		Point2d it = positions[i];
		Feature f;

		for (int i = 0, bitpos = 0; i < 512; i += 2, ++bitpos)
		{		
			int x1 = it.x + xp[i]; int y1 = it.y + yp[i];
			int x2 = it.x + xp[i + 1]; int y2 = it.y + yp[i + 1];
			f.setbit(bitpos, image[y1][x1] > image[y2][x2]);
		}
		f.position = (it);
		features[i] = f;
	}
	return features;

}
