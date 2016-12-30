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

using namespace cv;
using namespace af;
using namespace std;


#include <sstream>
#include <fstream>
#include "BriefPattern.h"




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
		cap.open("C:\\Users\\timya\\Videos\\Captures\\7.mp4");
	}
	if (!cap.isOpened())
		return -1;

	int video_mode = 1;

	Mat frame;
	if(video_mode==0)
		frame = imread("C:\\Users\\timya\\Desktop\\sign shapes.jpg");
	else
		if (!cap.read(frame))return;

	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);
	Mat out = Mat(frame.rows, frame.cols, CV_8UC1);
	cuArray<uchar> buf1(frame.cols*frame.rows);
	cuArray<uchar> bufx(frame.cols*frame.rows);
	cuArray<uint4> AngleMap(50000);
	
	Mat brief = Mat(320, 320, CV_8UC1);
	namedWindow("output",WINDOW_OPENGL);

	for (;;)
	{
		if(video_mode>0)
			if (!cap.read(frame))break;
		int points = 0;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		//medianBlur(grey, out, 3);

		buf1.upload(grey.data);
		profiler.Start();
		{
			FAST <<< dim3(frame.cols / FAST_TILE, frame.rows / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >>> (buf1, bufx,15, grey.cols, grey.rows);
		}
		std::stringstream ss;
		ss << "frame: " << profiler.Count() << " | " << profiler.Stop() <<"us | ";		
		bufx.download(out.data);

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
		AngleMap.upload(corners.data(), points);
		FAST_Refine << < corners.size() / 64, 64 >> > (buf1, AngleMap, corners.size(), grey.cols, grey.rows);
		AngleMap.download(corners.data(), points);
		
		
		for (int i = 0; i < points; ++i)
		{
			circle(frame, Point2d(corners[i].x, corners[i].y), 3, Scalar(255, 0, 255, 255));
		}

		ss << points<< " Corners";
		cv::putText(frame, ss.str(), Point2d(grey.cols / 2 - 500, grey.rows / 5 * 4), HersheyFonts::FONT_HERSHEY_DUPLEX, 2, Scalar(255,255, 255, 255));
	  	cv::imshow("output", frame);


		if (video_mode == 0)
		{
			waitKey();
			break;
		}
		else if (waitKey(1) >= 0) break;
	}
	return 0;
}