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
float focal = 600;
int slider;
void on_trackbar(int, void*)
{
	focal = (float)slider/300;
}

int main(int argc, char** argv)
{
	//if (argc < 2)return 0;
	//cudaSetDevice(1);
	Profiler::Enable();
	Orb orb = Orb::fromFile("C:\\Users\\timya\\Desktop\\Orb\\x64\\Release\\pat.txt",Orb::MODE_BRIEF);
	Profiler profiler;
	VideoCapture cap0,cap1;
	Mat L,R;
	namedWindow("frame", CV_WINDOW_NORMAL);
	//resizeWindow("frame", 1920, 1080);
	cap0.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cap0.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap1.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cap1.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap0.open(0);
	cap1.open(1);
	createTrackbar("focal", "frame", &slider, 1200, on_trackbar);
	if (!cap0.isOpened() || !cap1.isOpened())
		return -1;
	vector<Orb::Feature> features_old;
	Mat R_f, t_f;
	for (int fc = 0; waitKey(1) == -1; ++fc)
	{
		//if (!cap.read(frame))break;
		cap0.read(L);
		cap1.read(R);
		imshow("L", L);
		imshow("R", R);

		vector<Orb::Feature> featuresL = TrackKeypoints(L, orb, 2048);
		vector<Orb::Feature> featuresR = TrackKeypoints(R, orb, 2048);
		auto mps = ty::BRIEF::matchFeatures_gpu(featuresL, featuresR, 35);
		Mat fr(1080,1920, L.type());
		const float focal_length = 300;
		const float baseline = 1920;
		
		for (auto p : mps)
		{
			float distance = p.first.x - p.second.x;
			float z = focal*baseline / distance;
			float hue = z / focal_length * 255;
			circle(fr, p.first, 1, Scalar(255- hue,0, hue), 2, LINE_AA);

		}
	/*	for (auto f : featuresL)
		{
			circle(fr, f.position, 1, Scalar(12, 255, 128), 1, LINE_AA);
		}
		for (auto f : featuresR)
		{
			circle(fr, f.position + Point2f(L.cols, 0), 1, Scalar(12, 255, 128), 1, LINE_AA);
		}
		int inlier = 0;
		for (auto p : mps)
		{
			line(fr, p.first, p.second + Point2f(L.cols, 0), Scalar((1 - (float)p.second.y / (float)R.rows) * 255, ((float)p.second.y / (float)R.rows) * 255, ((float)p.second.y / (float)R.rows) * 255), 1, LINE_AA);
		}*/
		on_trackbar(slider, 0 );
		imshow("frame", fr);
	}
}