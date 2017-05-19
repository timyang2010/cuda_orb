#include <opencv2/opencv.hpp>
#include "Orb.h"
#include <vector>
using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	Orb orb;
	ty::Optimizer optimizer;
	vector<Orb::Feature> features;
	optimizer.generateTests(31, 5, 4);
	for (int i = 1; i<argc; ++i)
	{
		cout << argv[i] << endl;
		Mat m = imread(argv[i]);
		Mat grey;
		cvtColor(m, grey, CV_BGR2GRAY);
		uchar** grey2d = convert2D(grey.data, grey.cols, grey.rows);
		vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, 1500);
		optimizer.extractFeatures(grey2d, corners);
	}
	auto bts = optimizer.Optimize();
	Mat m = Mat::zeros(512, 512, CV_8UC1);
	for (auto t : bts)
	{
		line(m, Point2f(t.x1 * 16 + 256, t.y1 * 16 + 256), Point2f(t.x2 * 16 + 256, t.y2 * 16 + 256), Scalar(255), 1, cv::LINE_AA);
	}
	imshow("result", m);
	waitKey();
	optimizer.save("pat.txt", bts);
}
