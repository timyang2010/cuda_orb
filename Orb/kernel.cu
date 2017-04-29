
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include "Profiler.h"
#include "Memory.h"
#include "Orb.h"
using namespace cv;
using namespace std;


Mat renderTrajectory(Mat& iframe)
{
	const int hframe_count = 8;
	static vector<Mat> history;
	history.push_back(iframe);
	Mat rframe(iframe.rows, iframe.cols, CV_8UC1);
	if (history.size() > hframe_count)
	{
		history.erase(history.begin());
		for (int i = 0; i < hframe_count; ++i)
		{
			rframe += history[i]/(hframe_count-i);
		}
	}

	return rframe;
}

vector<Orb::Feature> TrackKeypoints(Mat& frame,Orb& orb)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	uchar** grey2d = convert2D(grey.data, grey.cols, grey.rows);
	vector<float4> corners = orb.detectKeypoints(grey, 25, 12, 1500);
	vector<Orb::Feature> features = orb.extractFeatures(grey2d, corners);
	return features;
}


int main(int argc,char** argv)
{
	Orb orb;
	BRIEF::Optimizer optimizer;
	Profiler profiler;
	VideoCapture cap; 	
	namedWindow("traj", WINDOW_NORMAL);
	resizeWindow("traj", 1280, 720);
	moveWindow("traj", 50, 50);
	cap.open(argv[1]);
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	Mat tframe = Mat(frame.rows,frame.cols,frame.type());
	vector<Orb::Feature> features_old;
	for (int fc = 0; waitKey(1)==-1; ++fc)
	{
		if (!cap.read(frame))break;
		vector<Orb::Feature> features = TrackKeypoints(frame, orb);
		for (vector<Orb::Feature>::iterator it = features.begin(); it < features.end(); ++it)
		{
			circle(frame, it->position, 2, Scalar(155, 255, 125), 1);
		}
		BRIEF::MultiLSHashTable hs;
		hs.InsertRange(features);
		if (features_old.size() > 0)
		{
			for (auto mp : hs.Hash_Match(features_old))
				line(frame, mp.first, mp.second, Scalar(255, 255, 225), 1, cv::LineTypes::LINE_AA);
		}
		features_old = features;
		imshow("traj",frame);
	}
	return 0;
}