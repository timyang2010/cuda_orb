#include <opencv2/opencv.hpp>
#include "Orb.h"
#include <vector>
#include "Profiler.h"
using namespace cv;
using namespace std;


Mat renderTrajectory(Mat& iframe)
{
	const int hframe_count = 8;
	static vector<Mat> history;
	history.push_back(iframe);
	Mat rframe(iframe.rows, iframe.cols, iframe.type());
	if (history.size() > hframe_count)
	{
		history.erase(history.begin());
		for (int i = 0; i < hframe_count; ++i)
		{ 
			rframe += history[i] / (hframe_count - i);
		}
	}
	return rframe;
}

vector<Orb::Feature> TrackKeypoints(Mat& frame, Orb& orb, int max_keypoints = 1000)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, max_keypoints);
	Profiler::global.Start();
	vector<Orb::Feature> features = orb.extractFeatures(grey, corners);
	Profiler::global.Log("BRIEF");
	return features;
}

int main(int argc, char** argv)
{
	if (argc < 2)return 0;

	Profiler::Enable();
	Orb orb = Orb::fromFile("C:\\Users\\timya\\Desktop\\Orb\\x64\\Release\\pat.txt");
	Profiler profiler;
	VideoCapture cap;
	Mat frame;
	namedWindow("traj", WINDOW_NORMAL);
	resizeWindow("traj", 1280, 720);
	moveWindow("traj", 50, 50);
	cap.open(argv[1]);
	if (!cap.isOpened())
		return -1;
	vector<Orb::Feature> features_old;
	for (int fc = 0; waitKey(1) == -1; ++fc)
	{
		if (fc % 2 == 0)continue;
		if (!cap.read(frame))break;
		Mat tframe(frame.rows, frame.cols, CV_8UC1);
		Mat grey;
		cvtColor(frame, grey, CV_BGR2GRAY);
		vector<Orb::Feature> features = TrackKeypoints(frame, orb);
		
		if (features_old.size() > 0)
		{
			auto mps = ty::MatchBF(features, features_old, 35);
			Profiler::global.Log("Matching");
			for (auto mp : mps)
			{
				if (pow(mp.first.x - mp.second.x, 2) + pow(mp.first.y - mp.second.y, 2)<10000)
					line(tframe, mp.first, mp.second, Scalar(255, 255, 225), 1, cv::LineTypes::LINE_AA);
			}

		}
		Profiler::global.Report();
		features_old = features;
		imshow("traj", grey+renderTrajectory(tframe));
	}
}