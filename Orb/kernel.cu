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


int main(int argc,char** argv)
{
	BRIEF extractor;
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
		
		FAST <<< dim3(frame.cols / FAST_TILE, frame.rows / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >>> (gpuInputBuffer, gpuOutputBuffer,10, grey.cols, grey.rows);

		
		gpuOutputBuffer.download(out.data);
		profiler.Log("FAST");
		vector<uint4> corners;
		for (uint i = 50; i < grey.cols - 50; ++i)
			for (uint j = 50; j < grey.rows -50; ++j)
			{
				uint cvalue = out.data[i + j*grey.cols];
				if (cvalue > 0)
				{					
					corners.push_back(uint4{ i,j,0,0});
					points++;
				}
			}
		int processed_count = points > FAST_Corner_Limit ? FAST_Corner_Limit : points;
		profiler.Start();
		AngleMap.upload(corners.data(), processed_count);
		FAST_Refine << < corners.size() / 32, 32 >> > (gpuInputBuffer, AngleMap, corners.size(), grey.cols, grey.rows);
		AngleMap.download(corners.data(), processed_count);	

		profiler.Log("Supression");
		std::sort(corners.begin(), corners.end(), [](const uint4&x, const uint4& y) {
			return x.w > y.w;
		});
		profiler.Log("Sort");
		int cc = min(processed_count, 2000);
		AngleMap.upload(corners.data(), cc);
		ComputeOrientation << < corners.size() / 32, 32 >> > (gpuInputBuffer, AngleMap, cc, grey.cols, grey.rows);
		AngleMap.download(corners.data(), cc);

		profiler.Log("Angle");
		

		vector<Point2d> poi;
		vector<float> angles;
		for (int i = 0; i < cc; ++i)
		{			

			if (corners[i].w > 1000000)
			{
				poi.push_back(Point2d(corners[i].x, corners[i].y));
				angles.push_back(corners[i].z);
				cv::circle(frame, Point2d(corners[i].x, corners[i].y), 1, Scalar(255, 255, 0, 0));
			}

		}
		profiler.Log("Draw");
		BRIEF::Features features = extractor.extractFeature(grey2d, poi, grey.cols, grey.rows);
		profiler.Log("rBRIEF");


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




		profiler.Report(); 
		
	  	cv::imshow("output", frame);
		if (waitKey(1) >= 0) break;
	}
	return 0;
}