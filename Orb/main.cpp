
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include "Profiler.h"
#include "Memory.h"
#include "Orb.h"
#include <sstream>
#include <fstream>
#include "Application.h"
#define _USE_MATH_DEFINES 
#include <math.h>
using namespace cv;
using namespace std;

Point2f _rotate(Point2f p, double deg, Point2f center)
{
	p = p - center;
	double _cos = cos(deg / 180 * M_PI);
	double _sin = sin(deg / 180 * M_PI);
	return Point2f(p.x*_cos - p.y*_sin, p.x*_sin + p.y*_cos) + center;
}

void match_keypoints(string n1, string n2)
{
	Orb orb = Orb::fromFile("pat.txt");
	Mat m1 = imread(n1);
	Mat m2 = imread(n2);
	auto f1 = TrackKeypoints(m1, orb, 1000);
	auto f2 = TrackKeypoints(m2, orb, 1000);
	auto pairs = ty::MatchBF(f1, f2, 45);
	Mat fr(max(m1.rows,m2.rows), m1.cols + m2.cols, m1.type());
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
		line(fr, p.first, p.second + Point2f(m1.cols, 0), Scalar((1-(float)p.second.y / (float)m2.rows) * 255, ((float)p.second.y / (float)m2.rows) * 255, ((float)p.second.y / (float)m2.rows) * 255), 1, LINE_AA);
	}
	imshow("match", fr);
	waitKey();
}

vector<float> rotate_test(string n1,Orb& orb,int max_distance)
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
		auto pairs = ty::MatchBF(f1, f2, max_distance);
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
void experiment(int argc, char** argv)
{
	
	Orb orb1 = Orb::fromFile("pat.txt");
	Orb orb2 = Orb();
	Mat gr(512, 512, CV_8UC1);
	Mat gr2(512, 512, CV_8UC1);
	for (int i = 0; i < 256; ++i)
	{
		cout << orb1[0][i] << endl;
		line(gr, Point2d(orb1[0][i].x1 * 6 + 256, orb1[0][i].y1 * 6 + 256), Point2d(orb1[0][i].x2 * 6 + 256, orb1[0][i].y2 * 6 + 256), Scalar(225), 1, LINE_AA);
		line(gr2, Point2d(orb2[0][i].x1 * 6 + 256, orb2[0][i].y1 * 6 + 256), Point2d(orb2[0][i].x2 * 6 + 256, orb2[0][i].y2 * 6 + 256), Scalar(225), 1, LINE_AA);
	}

	imshow("orb1",gr);
	imshow("orb2", gr2);
	waitKey();

	vector<vector<float>> crvector(120);

	for (int md = 35; md <= 50; md += 15)
	{
		vector<vector<float>> r1;
		vector<vector<float>> r2;
		vector<vector<float>> r3;
		for (int i = 2; i < argc; ++i)
		{
			r1.push_back(rotate_test(argv[i], orb2, md));
			r2.push_back(rotate_test(argv[i], orb2, md));
			r3.push_back(rotate_test(argv[i], orb1, md));
		}
		vector<float> v1 = vector_reduce_mean(r1);
		vector<float> v2 = vector_reduce_mean(r2);
		vector<float> v3 = vector_reduce_mean(r3);
		for (int i = 0; i < 120; ++i)
		{
			crvector[i].push_back(v1[i]);
			crvector[i].push_back(v2[i]);
			crvector[i].push_back(v3[i]);
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
}


int main(int argc,char** argv)
{
	switch (argv[1][0])
	{
	case 't':
		BRIEF_Optimize(argc,argv);
		break;
	case 'c':
		TrackCamera(string(argv[2]));
		break;
	case 'r':
		experiment(argc, argv);
		break;
	case 'm':
		match_keypoints(string(argv[2]), string(argv[3]));
		break;
	}
	return 0;
}