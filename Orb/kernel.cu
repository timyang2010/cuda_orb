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
	static unsigned int resultMapHost[2000 * 2000];
	SteeredBRIEF extractor;
	Profiler profiler;
	VideoCapture cap; 	

	
	for (int i = 0; i < 12; ++i)
	{
		Mat aux = Mat::zeros(512, 512, CV_8UC1);
		int* xp = extractor.getx(i);
		int* yp = extractor.gety(i);

		for (int j = 0; j < 512; j += 2)
			line(aux, Point2d(256+16*xp[j], 256 + 16 * yp[j]), Point2d(256 + 16 * xp[j + 1], 256 + 16 * yp[j + 1]), Scalar(255, 255, 255), 1);

		stringstream ss;
		ss << "+" << i*30;
		imshow(ss.str(), aux);
		
	}
	cap.open("C:\\Users\\timya\\Videos\\Captures\\Battlefield™ 1 1_3_2017 6_12_16 PM.mp4");
	if (!cap.isOpened())
		return -1;
	Mat frame;
	cap.read(frame);
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);
	Mat out = Mat(frame.rows, frame.cols, CV_8UC1);
	cuArray<uchar> gpuInputBuffer(frame.cols*frame.rows);
	cuArray<uchar> gpuOutputBuffer(frame.cols*frame.rows);
	cuArray<uint4> AngleMap(FAST_Corner_Limit);
	namedWindow("output");

	for (int fc=0;;++fc)
	{
		if (!cap.read(frame))break;
		int points = 0;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		gpuInputBuffer.upload(grey.data);
		profiler.Start();
		{
			FAST <<< dim3(frame.cols / FAST_TILE, frame.rows / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >>> (gpuInputBuffer, gpuOutputBuffer,10, grey.cols, grey.rows);
		}

		std::stringstream ss;
		ss << "frame: " << profiler.Count() << " | " << profiler.Stop() <<"us | ";		
		gpuOutputBuffer.download(out.data);

		vector<uint4> corners;
		for (uint i = 0; i < grey.cols; ++i)
			for (uint j = 0; j < grey.rows; ++j)
			{
				uint cvalue = out.data[i + j*grey.cols];
				if (cvalue > 0)
				{
					corners.push_back(uint4{ i,j,0,0});
					points++;
				}
			}
		int processed_count = points > FAST_Corner_Limit ? FAST_Corner_Limit : points;
		AngleMap.upload(corners.data(), processed_count);
		FAST_Refine << < corners.size() / 32, 32 >> > (gpuInputBuffer, AngleMap, corners.size(), grey.cols, grey.rows);
		AngleMap.download(corners.data(), processed_count);	
		std::sort(corners.begin(), corners.end(), [](const uint4&x, const uint4& y) {
			return x.w > y.w;
		});

		vector<Point2d> poi;
		for (int i = 0; i < min(processed_count,1400); ++i)
		{			
			if (i < 1400 )
			{
				poi.push_back(Point2d(Point2d(corners[i].x, corners[i].y)));
			}	
			else
			{
				ss << i << " Corners";
				break;
			}		
		}
		BRIEF::Features features;
		profiler.Start();
		extractor.extractFeature(grey.data, poi, features, grey.cols, grey.rows);
		cout << profiler.Stop() << endl;




		cv::putText(frame, ss.str(), Point2d(grey.cols / 2 - 500, grey.rows / 6 * 5), HersheyFonts::FONT_HERSHEY_DUPLEX, 2, Scalar(255,255, 255, 255));
	  	cv::imshow("output", frame);
		if (waitKey(1) >= 0) break;
	}
	return 0;
}