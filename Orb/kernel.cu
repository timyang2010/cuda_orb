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

int main(int argc,char** argv)
{
	rBRIEF extractor;
	Orb orb;
	Profiler profiler;
	VideoCapture cap; 	

	cap.open("\\\\140.118.7.213\\Dataset\\sequence\\13.mp4");
	if (!cap.isOpened())
	{
		return -1;
	}

	Mat frame;
	cap.read(frame);
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);
	
	int frameWidth = frame.cols, frameHeight = frame.rows;
	int frameSize = frameWidth*frameHeight;
	Mat trajectory( frameHeight,frameWidth, CV_8UC1);
	uchar** grey2d = convert2D(grey.data, frameWidth, frameHeight);
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
		vector<float4> corners;
		FAST(gpuInputBuffer,gpuOutputBuffer, corners , frameWidth, frameHeight);
		//AFFAST(grey, corners);
		profiler.Log("FAST");

		profiler.Start();
		orb.computeOrientation(gpuInputBuffer, corners, frameWidth, frameHeight);
		profiler.Log("Angle");
		int ang_hist[30] = { 0 };
		vector<float> angles;
		vector<Point2d> poi;
		for (int i = 0; i < corners.size(); ++i)
		{
			ang_hist[int(corners[i].z)]++;
			angles.push_back(corners[i].z);
			poi.push_back(Point2d(corners[i].x, corners[i].y));
		}
		for (int i = 0; i < 30; ++i)
		{
			cv::rectangle(frame, Rect2f(Point2f(50 , 500+(i - 15) * 20), Point2f(50 + ang_hist[i] / 2, 500 + (i - 15) * 20 + 19)), Scalar(255, 0, 255, 0), 1);
		}
		profiler.Log("Render");
		BRIEF::Features features = extractor.extractFeature(grey2d, poi, angles, grey.cols, grey.rows);
		//BRIEF::Features features = extractor.extractFeature(grey2d, poi, grey.cols, grey.rows);
		//BRIEF::Features features = extractor.extractFeature(grey2d, poi, grey.cols, grey.rows);
		profiler.Log("BRIEF");


		Mat hf(frameHeight, frameWidth, CV_8UC1);
		if (features_old.size() > 0)
		{
#pragma omp parallel 
			for (int i = 0; i < features.size(); ++i)
			{
				unsigned int min = INT_MAX;
				BRIEF::Feature f;
				Point2d pos;
				BRIEF::Feature p = features[i];
				for (int j = 0; j < features_old.size(); ++j)
				{
					unsigned int distance = features[i] - features_old[j];
					unsigned int L2 = pow(p.position.x - features_old[j].position.x, 2) + pow(p.position.y - features_old[j].position.y, 2);
					if (distance < 50 && L2<10000)
					{
						distance *= L2;
						if (distance < min)
						{
							min = distance;
							pos = features_old[j].position;
						}
					}
				}
				if (min < INT_MAX)
				{
					line(hf, features[i].position, pos, Scalar(255, 255, 0), 1, cv::LineTypes::LINE_AA);
					circle(hf, pos, 2, Scalar(0, 255, 0));
				}

				else
				{
					//circle(trajectory, Point2d(p.position), 2, Scalar(0, 0, 255));
				}
			}
		}

		profiler.Log("Match");
		features_old = features;


		profiler.Start();
		history.push_back(hf);
		Mat rframe(frameHeight, frameWidth, CV_8UC1);
		if (history.size() > 5)
		{
			history.erase(history.begin());
			
			for (int i = 0; i < 5; ++i)
			{
				rframe += history[i];
			}
		}
		profiler.Log("Render2");



		profiler.Report(); 
	  	cv::imshow("output", frame);	
		cv::imshow("traj", rframe);
		//waitKey();
		if (waitKey(1) >= 0) break;
	}
	return 0;
}