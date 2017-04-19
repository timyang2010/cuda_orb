#include <windows.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <arrayfire.h> 
#include "Profiler.cuh"
#include "Memory.cuh"
#include "FAST.cuh"
#include "Utility.cuh"
#include "brief.h"
#include <math.h>

using namespace cv;
using namespace af;
using namespace std;


#include <sstream>
#include <fstream>

#define Harris_Threshold 50000000
#define FAST_Corner_Limit 500000

void AFFAST(Mat& grey, vector<uint4>& poi)
{
	af::array afa = transpose(af::array(grey.cols, grey.rows, grey.data, af::source::afHost));
	af::features fast_features = af::fast(afa, 30, 12, true, 0.01f);
	int N = fast_features.getNumFeatures();
	af::array x_pos = fast_features.getX();
	af::array y_pos = fast_features.getY();
	af::array scores = fast_features.getScore();
	float* x = x_pos.host<float>();
	float* y = y_pos.host<float>();
	//float* x = x_pos.host<float>();
	for (int i = 0; i < N; ++i)
	{
		
		poi.push_back(uint4{ (uint)x[i],(uint)y[i], 0,0});
		//circle(frame, Point(x[i], y[i]), 2, Scalar(255, 0, 255), 1);
	}

}

int main(int argc,char** argv)
{
	BRIEF extractor;
	Profiler profiler;
	VideoCapture cap; 	

	cap.open("C:\\Users\\timya\\Videos\\Captures\\3.mp4");
	if (!cap.isOpened())
	{
		cout << "Load Failed" << endl;
		int p;
		cin >> p;
		return -1;

	}
	
	
	Mat frame;
	cap.read(frame);
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);

	uchar** grey2d = convert2D(grey.data, frame.cols, frame.rows);
	Mat integral = Mat(frame.rows, frame.cols, CV_32SC1);
	Mat out = Mat(frame.rows, frame.cols, CV_8UC1);
	cuArray<uchar> gpuInputBuffer(frame.cols*frame.rows);
	cuArray<uchar> gpuOutputBuffer(frame.cols*frame.rows);
	cuArray<uint4> AngleMap(FAST_Corner_Limit);
	namedWindow("output");
	BRIEF::Features features_old;
	for (int fc=0;;++fc)
	{
		if (!cap.read(frame))break;
		int points = 0;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		gpuInputBuffer.upload(grey.data);
		profiler.Start();
		vector<uint4> corners;
		AFFAST(grey, corners);
		for (int i = 0; i < corners.size(); ++i)
		{
			cv::circle(frame, Point2f(corners[i].x,corners[i].y), 2, Scalar(255, 255, 0, 0));
		}
		profiler.Log("FAST");

		int cc = corners.size() < 5000 ? corners.size() : 5000;
		AngleMap.upload(corners.data(), cc);
		ComputeOrientation << < corners.size() / 32, 32 >> > (gpuInputBuffer, AngleMap, cc, grey.cols, grey.rows);
		AngleMap.download(corners.data(), cc);

		profiler.Log("Angle");

		for (int i = 0; i < corners.size(); ++i)
		{
			cv::circle(frame, Point2f(corners[i].x, corners[i].y), 2, Scalar(255, 255, 0, 0));
			cv::line(frame, 
				Point2f(corners[i].x,corners[i].y), 
				Point2f(corners[i].x + 10*cos((float)corners[i].z/180*3.16159), corners[i].y + 10 * sin((float)corners[i].z / 180 * 3.16159)),
				Scalar(255, 255, 0, 0));
		}


	/*	BRIEF::Features features = extractor.extractFeature(grey2d, corners, grey.cols, grey.rows);
		profiler.Log("BRIEF");


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
					if (distance < 50 && L2<2500)
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
					line(frame, features[i].position, pos, Scalar(255, 255, 0), 1, cv::LineTypes::LINE_AA);
					circle(frame, pos, 2, Scalar(0, 255, 0));
				}

				else
				{
					circle(frame, Point2d(p.position), 2, Scalar(0, 0, 255));
				}
			}
		}

		features_old = features;

		profiler.Log("Match");*/


		profiler.Report(); 
		
	  	cv::imshow("output", frame);
		waitKey();
		//if (waitKey(1) >= 0) break;
	}
	return 0;
}