
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include "Profiler.h"
#include "Memory.h"
#include "Orb.h"

#include <sstream>
#include <fstream>
#include "Tracking.h"
using namespace cv;
using namespace std;

void BRIEF_Optimize(char* path_dict)
{	
	string path;
	fstream f(path_dict);
	Orb orb;
	BRIEF::Optimizer optimizer;
	vector<Orb::Feature> features;
	optimizer.generateTests();
	for (; getline(f, path);)
	{
		cout << path << endl;
		Mat m = imread(path);
		Mat grey;
		cvtColor(m, grey, CV_BGR2GRAY);
		uchar** grey2d = convert2D(grey.data, grey.cols, grey.rows);
		vector<float4> corners = orb.detectKeypoints(grey, 25, 12, 1000);
		vector<Point2f> poi;
		vector<float> angles;
		for (auto c : corners)
		{
			poi.push_back(Point2f(c.x, c.y));
			angles.push_back(c.z);
		}
		optimizer.extractFeatures(grey2d, poi, angles);
	}
	f.close();
	auto bts = optimizer.Optimize();
	Mat m = Mat::zeros(512, 512, CV_8UC1);
	for (auto t : bts)
	{
		line(m, Point2f(t.x1*16+256, t.y1 * 16 + 256), Point2f(t.x2 * 16 + 256, t.y2 * 16 + 256), Scalar(255), 1, cv::LINE_AA);
		cout << (int)t.x1<<" "<< (int)t.y1 << "   " << (int)t.x2 << " " << (int)t.y2 << endl;
	}
	imshow("result", m);
	waitKey();
}

int main(int argc,char** argv)
{
	string path;
	switch (argv[1][0])
	{
	case 't':
		BRIEF_Optimize(argv[2]);
		break;
	case 'c':
		fstream f(argv[2]);
		for (; getline(f, path);)
		{
			TrackCamera(path,loadPattern("C:\\Users\\timya\\Desktop\\pat.txt"));
		}
		break;

	}
	return 0;
}