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

void plotPattern(Mat& plot)
{
	
	for (int i = 0; i < 512; i+=2)
	{
		line(plot, Point2d(brief_x_pattern[i]*10, brief_y_pattern[i]*10), Point2d(brief_x_pattern[i+1] * 10, brief_y_pattern[i+1] * 10), Scalar(255, 255, 255));
	}
	
}

class brief_discriptor
{
public:
	brief_discriptor(int _x,int _y)
	{
		x = _x; y = _y;
	}
	union
	{
		unsigned int bitstring[8];
		__m256 vect;
	};
	int distance_to(brief_discriptor& discriptor)
	{

		for (int i = 0; i < 8; ++i)
		{
			unsigned int hdist = discriptor.bitstring[i] ^ bitstring[i];
		}
		return 0;
	}
	int x, y;
};

void cpuBRIEF(Mat& src, vector<uint4> cs,vector<brief_discriptor>& discriptors)
{
	for (vector<uint4>::iterator it = cs.begin(); it != cs.end(); ++it)
	{



	}
}



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
		cap.open("C:\\Users\\timya\\Videos\\Captures\\3.mp4");
	}
	if (!cap.isOpened())
		return -1;

	int video_mode = 0;

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
		buf1.upload(grey.data);
		profiler.Start();
		{
			FAST << <dim3(frame.cols / FAST_TILE, frame.rows / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >> > (buf1, bufx, 35, grey.cols, grey.rows);
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
					corners.push_back(uint4{ i,j,0,0 });
					points++;
				}
			}
		AngleMap.upload(corners.data(), points);
		OrientFast << <(points / 64 + 1), 64 >> > (buf1, AngleMap, grey.cols, grey.rows, points);
		AngleMap.download(corners.data(), points);
		
		
		for (int i = 0; i < points; ++i)
		{
			circle(frame, Point2d(corners[i].x, corners[i].y), 3, Scalar(255, 0, 255, 255));
			//line(frame, Point2d(corners[i].x, corners[i].y), Point2d(corners[i].w, corners[i].z), Scalar(0, 0, 255), 1,cv::LineTypes::LINE_AA);
		}

		ss << points<< " Corners";
		cv::putText(frame, ss.str(), Point2d(grey.cols / 2 - 500, grey.rows / 5 * 4), HersheyFonts::FONT_HERSHEY_DUPLEX, 2, Scalar(255, 0, 255, 255));
	  	cv::imshow("output", frame);

		plotPattern(brief);
		imshow("bf", brief);
		if (video_mode == 0)
		{
			waitKey();
			break;
		}
		else if (waitKey(1) >= 0) break;
	}
	return 0;
}