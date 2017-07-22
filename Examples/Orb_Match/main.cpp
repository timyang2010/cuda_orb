#include <opencv2/opencv.hpp>
#include "Orb.h"
#include <vector>
using namespace cv;
using namespace std;

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
	Orb orb = Orb::fromFile("C:\\Users\\timya\\Desktop\\Orb\\x64\\Release\\pat.txt");
	if (argc < 3)return -1;
	Mat m1 = imread(argv[1]);
	Mat m2 = imread(argv[2]);
	auto f1 = TrackKeypoints(m1, orb, 1000);
	auto f2 = TrackKeypoints(m2, orb, 1000);
	auto pairs = ty::BRIEF::matchFeatures(f1, f2, 45);
	Mat fr(max(m1.rows, m2.rows), m1.cols + m2.cols, m1.type());
	m1.copyTo(fr(Rect2d(0, 0, m1.cols, m1.rows)));
	m2.copyTo(fr(Rect2d(m1.cols, 0, m2.cols, m2.rows)));
	for (auto f : f1)
	{
		circle(fr, f.position, 1, Scalar(12, 255, 128), 1, LINE_AA);
	}
	for (auto f : f2)
	{
		circle(fr, f.position + Point2f(m1.cols, 0), 1, Scalar(12, 255, 128), 1, LINE_AA);
	}
	int inlier = 0;
	for (auto p : pairs)
	{
		line(fr, p.first, p.second + Point2f(m1.cols, 0), Scalar((1 - (float)p.second.y / (float)m2.rows) * 255, ((float)p.second.y / (float)m2.rows) * 255, ((float)p.second.y / (float)m2.rows) * 255), 1, LINE_AA);
	}
	imshow("match", fr);
	waitKey();
	return 0;
}
