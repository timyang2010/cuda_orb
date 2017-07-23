#include <opencv2/opencv.hpp>
#include "Orb.h"
#include <vector>
#include <iomanip>
#include "Profiler.h"
using namespace cv;
using namespace std;


Mat renderTrajectory(Mat& iframe)
{
	const int hframe_count = 2;
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
	cudaSetDevice(1);
	Profiler::Enable();
	Orb orb = Orb::fromFile("C:\\Users\\timya\\Desktop\\Orb\\x64\\Release\\pat.txt",Orb::MODE_BRIEF);
	Profiler profiler;
	VideoCapture cap;
	Mat frame;
	Mat trajectory(512, 512, CV_32FC4);
	namedWindow("traj", WINDOW_NORMAL);
	resizeWindow("traj", 1280, 720);
	moveWindow("traj", 50, 50);
	cap.open(argv[1]);
	if (!cap.isOpened())
		return -1;
	vector<Orb::Feature> features_old;
	Mat R_f, t_f;
	for (int fc = 0; waitKey(1) == -1; ++fc)
	{
		if (!cap.read(frame))break;
		//if (fc % 2 == 0)continue;
		Mat tframe(frame.rows, frame.cols, CV_8UC1);
		Mat grey;
		cvtColor(frame, grey, CV_BGR2GRAY);
		vector<Orb::Feature> features = TrackKeypoints(frame, orb,2048);

		if (features_old.size() > 0)
		{
			auto mps = ty::BRIEF::matchFeatures_gpu(features, features_old, 35);
			Profiler::global.Log("Matching");

			double focal = 560;
			cv::Point2d pp(1920/2, 1080/2);
			vector<Point2f> v1, v2;

			for (auto mp : mps)
			{
				int L2 = pow(mp.first.x - mp.second.x, 2) + pow(mp.first.y - mp.second.y, 2);
				if (L2 < 20000)
				{
					line(tframe, mp.first, mp.second, Scalar(255, 255, 255), 1, cv::LineTypes::LINE_AA);
					v1.push_back(mp.first);
					v2.push_back(mp.second);
				}

			}
			Profiler::global.Start();
			if (fc < 4)
			{
				Mat E, R, t, mask;
				E = findEssentialMat(v1, v2, focal, pp, RANSAC, 0.999, 1.0, mask);
				recoverPose(E, v1, v2, R, t, focal, pp, mask);
				R_f = R.clone();
				t_f = t.clone();
			}
			else
			{
				Mat E, R, t, mask;
				E = findEssentialMat(v1, v2, focal, pp, RANSAC, 0.999, 1.0, mask);
				recoverPose(E, v1, v2, R, t, focal, pp, mask);
				t_f = t_f + 1*(R_f*t);
				R_f = R*R_f;
				int x = int(t_f.at<double>(0)) + 256;
				int y = int(t_f.at<double>(2)) + 256;
				circle(trajectory, Point(x, y), 1, CV_RGB(255, 60, 0), 1);
			}
			Profiler::global.Log("Pose");
		}
		Profiler::global.Report();
		features_old = features;
		imshow("traj", grey+renderTrajectory(tframe));
		imshow("trajectory", trajectory);

	}
}