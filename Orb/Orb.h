#pragma once
#include "cuda_runtime.h"
#include "Memory.cuh"
#include "brief.h"
#include <vector>
#include <opencv2/opencv.hpp>
#define CORNER_LIMIT 500000
class Orb
{
public:
	Orb();
	void computeOrientation(cuArray<unsigned char>& frame, std::vector<float4>& corners, int width, int height);
	std::vector<float4> fast(cuArray<uchar>& ibuffer, cuArray<uchar>& aux, const int width, const int height, const int padding = 50);
private:
	cuArray<float4> AngleMap = cuArray<float4>(CORNER_LIMIT);
};



void AFFAST(cv::Mat& grey, std::vector<float4>& poi);