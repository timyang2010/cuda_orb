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
#include <opencv2\core\cuda.hpp>
#include <opencv2\features2d.hpp>

#include <math.h>

using namespace cv;
using namespace af;
using namespace std;


#include <sstream>
#include <fstream>

#define Harris_Threshold 50000000
#define FAST_Corner_Limit 500000

void AFFAST(cuArray<uchar>& ibuffer,cuArray<uchar>& aux, const int width, const int height, vector<float4>& poi, const int padding = 50)
{
	
	Mat auxmat = Mat(width, height, CV_8UC1);
	FAST << < dim3(width / FAST_TILE, height / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >> > (ibuffer, aux, 40, width, height);
	aux.download(auxmat.data);
	for (uint i = padding; i < width - padding; ++i)
		for (uint j = padding; j < height - padding; ++j)
		{
			uint cvalue = auxmat.data[i + j*width];
			if (cvalue > 0)
			{
				poi.push_back({ (float)i,(float)j,0,0 });
			}
		}

}

int main(int argc,char** argv)
{
	rBRIEF extractor;
	Profiler profiler;
	VideoCapture cap; 	

	cap.open("C:\\Users\\timya\\Desktop\\203394129.mp4");
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
	
	cuArray<uchar> gpuInputBuffer(frame.cols*frame.rows);
	cuArray<uchar> gpuOutputBuffer(frame.cols*frame.rows);
	cuArray<float4> AngleMap(FAST_Corner_Limit);
	namedWindow("output");
	BRIEF::Features features_old;
	for (int fc=0;;++fc)
	{
		if (!cap.read(frame))break;
		int points = 0;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		if (countNonZero==0)continue;
		gpuInputBuffer.upload(grey.data);
		profiler.Start();
		vector<float4> corners;
		AFFAST(gpuInputBuffer,gpuOutputBuffer,grey.cols,grey.rows, corners);
		
		profiler.Log("FAST");
		for (int i = 0; i < corners.size(); ++i)
		{
			cv::circle(frame, Point2f(corners[i].x, corners[i].y), 2, Scalar(255, 255, 0, 0));
		}
		profiler.Start();

		int cc = corners.size() < FAST_Corner_Limit ? corners.size() : FAST_Corner_Limit;
		AngleMap.upload(corners.data(), cc);
		ComputeOrientation << < corners.size() / 32, 32 >> > (gpuInputBuffer, AngleMap, cc, grey.cols, grey.rows);
		AngleMap.download(corners.data(), cc);
		profiler.Log("Angle");

		int ang_hist[12] = { 0 };
		vector<float> angles;
		vector<Point2d> poi;
		for (int i = 0; i < corners.size(); ++i)
		{
			cv::circle(frame, Point2f(corners[i].x, corners[i].y), 2, Scalar(255, 255, 0, 0));
			cv::line(frame, 
				Point2f(corners[i].x,corners[i].y), 
				Point2f(corners[i].x + 10*cos(corners[i].z), corners[i].y + 10 * sin(corners[i].z)),
				Scalar(255, 255, 0, 0));
			ang_hist[int(corners[i].z / 3.14159 * 180/6)]++;
			angles.push_back(0);
			poi.push_back(Point2d(corners[i].x, corners[i].y));
		}
		for (int i = 0; i < 12; ++i)
		{
			cv::rectangle(frame, Rect2f(Point2f(50 , 120+(i - 6) * 20), Point2f(50 + ang_hist[i] / 2, 120 + (i - 6) * 20 + 19)), Scalar(255, 0, 255, 0), 1);
		}
		profiler.Log("Render");
		//BRIEF::Features features = extractor.extractFeature(grey2d, poi, angles, grey.cols, grey.rows);
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
		
		//waitKey();
		if (waitKey(1) >= 0) break;
	}
	return 0;
}