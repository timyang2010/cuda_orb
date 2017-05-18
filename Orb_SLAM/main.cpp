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

vector<Orb::Feature> TrackKeypoints(Mat& frame, Orb& orb, int max_keypoints = 2000)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, max_keypoints);
	vector<Orb::Feature> features = orb.extractFeatures(grey, corners);
	return features;
}

int main(int argc, char** argv)
{
	if (argc < 2)return 0;
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
			ty::MultiLSHashTable hs;
			hs.InsertRange(features);
			for (auto mp : hs.Hash_Match(features_old, 64))
			{
				if (pow(mp.first.x - mp.second.x, 2) + pow(mp.first.y - mp.second.y, 2)<10000)
					line(tframe, mp.first, mp.second, Scalar(255, 255, 225), 1, cv::LineTypes::LINE_AA);
			}

		}
		features_old = features;
		imshow("traj", renderTrajectory(tframe));
	}
}