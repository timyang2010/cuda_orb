#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "Profiler.h"
#include "Memory.cuh"
#include "FAST.cuh"
#include <math.h>
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
			rframe += history[i];
		}
	}
	return rframe;
}

int main(int argc,char** argv)
{
	const int hframe_count = 9;
	const int padding = 50;
	rBRIEF extractor;
	Orb orb;
	Profiler profiler;
	VideoCapture cap; 	

	cap.open("\\\\140.118.7.213\\Dataset\\sequence\\3.mp4");
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	cap.read(frame);
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);	
	int frameWidth = frame.cols, frameHeight = frame.rows;
	int frameSize = frameWidth*frameHeight;
	Mat i5 = Mat(frame.rows, frame.cols, CV_8UC1);
	uchar** grey2d = convert2D(i5.data, frameWidth, frameHeight);
	cuArray<uchar> gpuInputBuffer(frameSize);
	cuArray<uchar> gpuOutputBuffer(frameSize);
	cuArray<float4> AngleMap(CORNER_LIMIT);
	vector<Mat> history;
	BRIEF::Features features_old;

	for (int fc=0;;++fc)
	{
		if (!cap.read(frame))break;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		gpuInputBuffer.upload(grey.data);
		

		profiler.Start();
		vector<float4> corners = orb.fast(gpuInputBuffer, gpuOutputBuffer, frameWidth, frameHeight);
		orb.computeOrientation(gpuInputBuffer, corners, frameWidth, frameHeight);
		profiler.Log("FAST+Orientation");

		profiler.Start();
		boxFilter(grey, i5, -1, Size(5, 5));
		profiler.Log("Blur");

		vector<Point2d> keypoints;
		vector<float> angles;
		for (int i = 0; i < corners.size(); ++i)
		{			
			keypoints.push_back(Point2d(corners[i].x, corners[i].y));
			angles.push_back(corners[i].z);
		}

		profiler.Log("Render");
		BRIEF::Features features = extractor.extractFeature(grey2d, keypoints, angles, grey.cols, grey.rows);
		profiler.Log("BRIEF");

		Mat hf(frameHeight, frameWidth, CV_8UC1);
		for (auto v: MatchBF(features, features_old))
			line(hf, v.first,v.second, Scalar(255, 255, 0), 1, cv::LineTypes::LINE_AA);

		profiler.Log("Match");
		features_old = features;
		profiler.Report(); 
	  	cv::imshow("output", frame);	
		cv::imshow("traj", renderTrajectory(hf));
		//waitKey();
		if (waitKey(1) >= 0) break;
	}
	return 0;
}