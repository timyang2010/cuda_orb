#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "Profiler.h"
#include "Memory.h"
#include <fstream>
#include "Application.h"
using namespace cv;
using namespace std;

void BRIEF_Optimize(int argc, char** argv)
{
	Orb orb;
	ty::Optimizer optimizer;
	vector<Orb::Feature> features;
	optimizer.generateTests(31,5,4);
	for (int i=2;i<argc;++i)
	{
		cout << argv[i] << endl;
		Mat m = imread(argv[i]);
		Mat grey;
		cvtColor(m, grey, CV_BGR2GRAY);
		uchar** grey2d = convert2D(grey.data, grey.cols, grey.rows);
		vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, 1500);
		vector<Point2f> poi;
		vector<float> angles;
		for (auto c : corners)
		{
			poi.push_back(Point2f(c.x, c.y));
			angles.push_back(c.z);
		}
		optimizer.extractFeatures(grey2d, poi, angles);
	}
	auto bts = optimizer.Optimize();
	Mat m = Mat::zeros(512, 512, CV_8UC1);
	fstream of("pat.txt",ios::out);
	for (auto t : bts)
	{
		line(m, Point2f(t.x1 * 16 + 256, t.y1 * 16 + 256), Point2f(t.x2 * 16 + 256, t.y2 * 16 + 256), Scalar(255), 1, cv::LINE_AA);
		cout << (int)t.x1 << " " << (int)t.y1 << "   " << (int)t.x2 << " " << (int)t.y2 << endl;
	}
	imshow("result", m);
	waitKey();
	for (auto t : bts)
	{
		of << (int)t.x1 << " " << (int)t.y1 << " " << (int)t.x2 << " " << (int)t.y2 << endl;
	}
	of.close();
}

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

vector<Orb::Feature> TrackKeypoints(Mat& frame, Orb& orb,int max_keypoints)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	uchar** grey2d = convert2D(grey.data, grey.cols, grey.rows);
	vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, max_keypoints);
	vector<Orb::Feature> features = orb.extractFeatures(grey2d, corners);
	return features;
}
void TrackCamera(string arg)
{
	Orb orb = Orb::fromFile("C:\\Users\\timya\\Desktop\\Orb\\x64\\Release\\pat.txt");
	Profiler profiler;
	VideoCapture cap;
	Mat frame;
	ty::Optimizer opt;
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
		if (features_old.size() > 0)
		{
			ty::MultiLSHashTable hs;
			hs.InsertRange(features);
			for (auto mp : hs.Hash_Match(features_old, 64))
			{
				if(pow(mp.first.x-mp.second.x,2)+pow(mp.first.y-mp.second.y,2)<10000)
				line(tframe, mp.first, mp.second, Scalar(255, 255, 225), 1, cv::LineTypes::LINE_AA);
			}
				
		}
		features_old = features;
		imshow("traj", grey / 2 + renderTrajectory(tframe));
	}
}
