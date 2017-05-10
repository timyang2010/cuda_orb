#pragma once
#ifndef ORB_H
#define ORB_H
#include "cuda_runtime.h"
#include "Memory.h"
#include "BRIEF.h"
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#define CORNER_LIMIT 100000
class Orb : public BRIEF::rBRIEF
{
public:
	enum MODE;
	Orb();
	Orb(int s, MODE mode = MODE_RBRIEF);
	Orb(std::vector<BRIEF::BinaryTest> tests, MODE mode = MODE_RBRIEF);
	//compute orientation of keypoints using intensity centroid method
	void computeOrientation(cuArray<uchar>& frame, std::vector<float4>& corners, int width, int height);	
	//detect keypoints using FAST and Harris threshold method, calling this function reduces the initialization overhead of cuArray<T>
	std::vector<float4> detectKeypoints(cuArray<uchar>& ibuffer, cuArray<uchar>& aux,int thres, const int arc_length, const int width, const int height, const int limit = 1000,const int padding = 45);
	//cv::Mat wrapper for detectKeypoints
	std::vector<float4> detectKeypoints(cv::Mat& frame, int thres, const int arc_length, const int limit = 1000, const int padding = (int)(BRIEF_DEFAULT_WINDOW_SIZE*0.8f));
	//compute rBRIEF descriptors given a set of keypoints
	std::vector<Feature> extractFeatures(uint8_t** image, std::vector<float4> keypoints) const;
	static Orb fromFile(char* filename);
	enum MODE
	{
		MODE_BRIEF,
		MODE_RBRIEF
	};
private:
	cuArray<float4> AngleMap = cuArray<float4>(CORNER_LIMIT);
	MODE _mode;
};




#endif