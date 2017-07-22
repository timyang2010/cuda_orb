#include <opencv2/opencv.hpp>
#include "Orb.h"
#include <vector>
using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES 
#include <math.h>

vector<Orb::Feature> TrackKeypoints(Mat& frame, Orb& orb, int max_keypoints = 2000)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, max_keypoints);
	vector<Orb::Feature> features = orb.extractFeatures(grey, corners);
	return features;
}

Point2f _rotate(Point2f p, double deg, Point2f center)
{
	p = p - center;
	double _cos = cos(deg / 180 * M_PI);
	double _sin = sin(deg / 180 * M_PI);
	return Point2f(p.x*_cos - p.y*_sin, p.x*_sin + p.y*_cos) + center;
}

void extract_patch(int argc, char** argv)
{
	const string path = "\\\\140.118.7.213\\Dataset\\pos1\\";
	ty::Optimizer optimizer;
	optimizer.generateTests(31, 5, 2, 1);
	for (int i = 1; i < 1200; ++i)
	{
		stringstream ss;
		ss << i << ".jpg";
		string p = path + ss.str();
		cout << i << endl;
		Mat patch = imread(path + ss.str(), CV_LOAD_IMAGE_GRAYSCALE);
		Mat normalized;
		resize(patch, normalized, Size2f(31, 31));
		ty::Keypoint k(16, 16);
		vector<ty::Keypoint> kps;
		kps.push_back(k);
		optimizer.extractFeatures(convert2D<uchar>(normalized.data, normalized.cols, normalized.rows), kps);
	}
	auto result = optimizer.Optimize(0.2, 0.01);
	Mat gr = Mat::zeros(512, 512, CV_8UC1);
	for (int i = 0; i < 256; ++i)
	{
		line(gr, Point2d(result[i].x1 * 16 + 256, result[i].y1 * 16 + 256), Point2d(result[i].x2 * 16 + 256, result[i].y2 * 16 + 256), Scalar(225), 1, LINE_AA);
	}
	imshow("x", gr);
	waitKey();
	optimizer.save("test", result);
}


vector<float> rotate_test(string n1, Orb& orb, int max_distance)
{

	Mat m1 = imread(n1);
	vector<float> match_result;
	for (int i = 0; i < 360; i += 3)
	{
		Mat rm2;
		Point2f center = Point2f(m1.cols / 2, m1.rows / 2);
		Mat rm = getRotationMatrix2D(center, i, 1);
		warpAffine(m1, rm2, rm, Size2d(m1.cols, m1.rows));
		auto f1 = TrackKeypoints(m1, orb, 100000);
		auto f2 = TrackKeypoints(rm2, orb, 100000);
		auto pairs = ty::BRIEF::matchFeatures(f1, f2, max_distance);
		cout << f1.size() + f2.size() << endl;
		Mat fr(m1.rows, m1.cols + m1.cols, m1.type());
		m1.copyTo(fr(Rect2d(0, 0, m1.cols, m1.rows)));
		rm2.copyTo(fr(Rect2d(m1.cols, 0, m1.cols, m1.rows)));
		int inlier = 0;
		for (auto p : pairs)
		{
			Point2f fp = _rotate(p.first, -i, center) - p.second;
			if (abs(fp.x) < 5 && abs(fp.y) < 5)
			{
				inlier++;
			}
		}
		match_result.push_back((double)inlier / (double)pairs.size());
	}
	return match_result;
}
vector<float>& vector_sum(vector<float>&v1, vector<float>&v2)
{
	for (int i = 0; i < v1.size(); ++i)
	{
		v1[i] += v2[i];
	}
	return v1;
}
vector<float> vector_reduce_mean(vector<vector<float>>& v)
{
	vector<float> tmp(v[0].size());
	for (int i = 0; i < v.size(); ++i)
	{
		vector_sum(tmp, v[i]);
	}
	for (int i = 0; i < tmp.size(); ++i)
	{
		tmp[i] /= v.size();
	}
	return tmp;
}
int main(int argc, char** argv)
{

	Orb orb1;
	Orb orb2 = Orb::fromFile("C:\\Users\\timya\\Desktop\\Orb\\x64\\Release\\pat2.txt");
	Mat gr(512, 512, CV_8UC1);
	Mat gr2(512, 512, CV_8UC1);
	for (int i = 0; i < 256; ++i)
	{
		cout << orb1[0][i] << endl;
		line(gr, Point2d(orb1[0][i].x1 * 16 + 256, orb1[0][i].y1 * 16 + 256), Point2d(orb1[0][i].x2 * 16 + 256, orb1[0][i].y2 * 16 + 256), Scalar(225), 1, LINE_AA);
		line(gr2, Point2d(orb2[0][i].x1 * 16 + 256, orb2[0][i].y1 * 16 + 256), Point2d(orb2[0][i].x2 * 16 + 256, orb2[0][i].y2 * 16 + 256), Scalar(225), 1, LINE_AA);
	}

	imshow("orb1", gr);
	imshow("orb2", gr2);
	imwrite("orb1.jpg", gr);
	imwrite("orb2.jpg", gr2);
	waitKey();

	vector<vector<float>> crvector(120);

	for (int md = 35; md <= 36; md += 15)
	{
		vector<vector<float>> r1;
		vector<vector<float>> r2;
		for (int i = 1; i < argc; ++i)
		{
			r1.push_back(rotate_test(argv[i], orb1, md));
			r2.push_back(rotate_test(argv[i], orb2, md));
		}
		vector<float> v1 = vector_reduce_mean(r1);
		vector<float> v2 = vector_reduce_mean(r2);
		for (int i = 0; i < 120; ++i)
		{
			crvector[i].push_back(v1[i]);
			crvector[i].push_back(v2[i]);
		}
		cout << md << endl;
	}

	ofstream f("log.txt");
	for (int i = 0; i < crvector.size(); ++i)
	{
		for (vector<float>::iterator it = crvector[i].begin(); it != crvector[i].end(); ++it)
		{
			f << *it << " ";
		}
		f << endl;
	}
	f.close();
	return 0;
}