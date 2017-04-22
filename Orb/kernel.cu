#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "Profiler.h"
#include "Memory.cuh"
#include "FAST.cuh"
#include "Orb.h"
#include <thread>
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

void blurmat(Mat& m)
{
	boxFilter(m, m, -1, Size(5, 5));
}
#define p_match pair<Point2f, Point2f>
vector<p_match> stddev_reject(vector<p_match>& matches)
{
	vector<p_match> filtered;
	double sum = 0,stddev = 0;
	vector<float> ang(matches.size());
	for (int i=0;i<matches.size();++i)
	{
		Point2f p = (matches[i].first - matches[i].second);		
		ang[i] = atan2(p.y, p.x);
		sum += ang[i];
	}
	sum /= matches.size();
	for (int i = 0; i < matches.size(); ++i)
	{
		stddev += sqrt(pow(ang[i] - sum,2));
	}
	for (int i = 0; i < matches.size(); ++i)
	{
		if (abs(ang[i] - sum) < stddev)
		{
			filtered.push_back(matches[i]);
		}
	}
	return filtered;
}

int main(int argc,char** argv)
{
	const int padding = 50;
	const string dir = "C:\\Users\\timya\\Desktop\\203394129.mp4";
	rBRIEF extractor;
	Orb orb;
	Profiler profiler;
	VideoCapture cap; 	
	namedWindow("traj", WINDOW_NORMAL);
	resizeWindow("traj", 1280, 720);
	moveWindow("traj", 50, 50);
	cap.open(dir);
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	cap.read(frame);
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);	
	int frameWidth = frame.cols, frameHeight = frame.rows;
	int frameSize = frameWidth*frameHeight;
	uchar** grey2d = convert2D(grey.data, frameWidth, frameHeight);
	cuArray<uchar> gpuInputBuffer(frameSize), gpuOutputBuffer(frameSize);
	BRIEF::Features features_old;
	
	for (int fc = 0; waitKey(1)==-1; ++fc)
	{
		if (!cap.read(frame))break;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		gpuInputBuffer.upload(grey.data);
		


		profiler.Start();	
		std::thread first([&]
		{
			boxFilter(grey, grey, -1, Size(5, 5));
		});
		vector<float4> corners = orb.fast(gpuInputBuffer, gpuOutputBuffer, 35, 9, frameWidth, frameHeight);
		orb.computeOrientation(gpuInputBuffer, corners, frameWidth, frameHeight);
		first.join();
		profiler.Log("FAST");

		vector<Point2d> keypoints;
		vector<float> angles;
		for (int i = 0; i < corners.size(); ++i)
		{			
			keypoints.push_back(Point2d(corners[i].x, corners[i].y)); 
			angles.push_back(corners[i].z);
		}

		profiler.Start();
		BRIEF::Features features = extractor.extractFeature(grey2d, keypoints, angles, grey.cols, grey.rows);
		profiler.Log("BRIEF");

		Mat hf(frameHeight, frameWidth, CV_8UC1);
		profiler.Start();
		
		MultiLSHashTable h;		
		h.InsertRange(features);	
		profiler.Log("Hash_Build");
		auto mpairs = h.Hash_Match(features_old,35);

		profiler.Log("Hash_Match");
		for (auto v : mpairs)
			line(hf, v.first, v.second, Scalar(255), 0.5, cv::LineTypes::LINE_AA);
		features_old = features;
		profiler.Log("Render");
		
		//cv::imshow("traj", grey+ renderTrajectory(hf));
		profiler.Log("Display");
		profiler.Report();
	}
	return 0;
}