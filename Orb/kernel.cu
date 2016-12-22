#include <windows.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <arrayfire.h>
#include "Profiler.cuh"
#include "Memory.cuh"
#include "FAST.cuh"
using namespace cv;
using namespace af;

#include <sstream>
int main(int argc,char** argv)
{
	Profiler profiler;
	VideoCapture cap; 	
	if (argc<2)
	{
		cap.open(0);
	}
	else
	{
		cap.open(argv[1]);
	}
	if (!cap.isOpened())
		return -1;

	
	Mat frame;
	if (!cap.read(frame))return;
		
	
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);
	Mat out = Mat(frame.rows, frame.cols, CV_8UC1);
	
	cuArray<uchar> buf1(frame.cols*frame.rows);
	cuArray<uchar> bufx(frame.cols*frame.rows);
	namedWindow("output",WINDOW_OPENGL);
	for (;;)
	{
		if (!cap.read(frame))break;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		buf1.upload(grey.data);
		profiler.Start();
		{
			FAST << <dim3(frame.cols / FAST_TILE, frame.rows / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >> > (buf1, bufx, 35, grey.cols, grey.rows);
		}
		std::stringstream ss;
		ss << "frame: " << profiler.Count() << " | " << profiler.Stop() <<"us | ";		
		bufx.download(out.data);
		int points = 0;
		for (int i = 0; i < grey.cols; ++i)
			for (int j = 0; j < grey.rows; ++j)
				if (out.data[i + j*grey.cols] > 0)
				{
					circle(frame, Point2d(i, j), 3, Scalar(255,0, 255, 255));
					points++;
				}	
		ss << points<< " Corners";
		cv::putText(frame, ss.str(), Point2d(grey.cols / 2 - 500, grey.rows / 5 * 4), HersheyFonts::FONT_HERSHEY_DUPLEX, 2, Scalar(255, 255, 0, 255));
	  	cv::imshow("output", frame);
		if (waitKey(1) >= 0) break;
	}
	return 0;
}