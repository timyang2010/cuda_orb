#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include "Profiler.h"
#include "Memory.h"
#include "Orb.h"
#include <sstream>
#include <fstream>
using namespace cv;
using namespace std;
Mat renderTrajectory(Mat& iframe)
{
	const int hframe_count = 8;
	static vector<Mat> history;
	history.push_back(iframe);
	Mat rframe(iframe.rows, iframe.cols, iframe.type());
	if (history.size() > hframe_count)
	{
		history.erase(history.begin());
		for (int i = 0; i < hframe_count; ++i)
		{
			rframe += history[i] / (hframe_count - i);
		}
	}
	return rframe;
}

vector<Orb::Feature> TrackKeypoints(Mat& frame, Orb& orb)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	uchar** grey2d = convert2D(grey.data, grey.cols, grey.rows);
	vector<float4> corners = orb.detectKeypoints(grey, 25, 12, 1500);
	vector<Orb::Feature> features = orb.extractFeatures(grey2d, corners, Orb::MODE::MODE_BRIEF);
	return features;
}
void TrackCamera(string arg, Orb orb)
{
	Profiler profiler;
	VideoCapture cap;
	Mat frame;
	BRIEF::Optimizer opt;
	namedWindow("traj", WINDOW_NORMAL);
	resizeWindow("traj", 1280, 720);
	moveWindow("traj", 50, 50);

	cap.open(arg);
	if (!cap.isOpened())
		return;
	vector<Orb::Feature> features_old;
	for (int fc = 0; waitKey(1) == -1; ++fc)
	{
		if (!cap.read(frame))break;
		Mat tframe(frame.rows, frame.cols, CV_8UC1);
		Mat grey;
		cvtColor(frame, grey, CV_BGR2GRAY);
		vector<Orb::Feature> features = TrackKeypoints(frame, orb);
		cout << opt.computeVariance(features);
		if (features_old.size() > 0)
		{
			BRIEF::MultiLSHashTable hs;
			hs.InsertRange(features);
			for (auto mp : hs.Hash_Match(features_old, 32))
				line(tframe, mp.first, mp.second, Scalar(255, 255, 225), 1, cv::LineTypes::LINE_AA);
		}
		features_old = features;

		imshow("traj", grey / 2 + renderTrajectory(tframe));
	}
}
