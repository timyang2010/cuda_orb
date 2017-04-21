#include "brief.h"
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
vector< pair<Point2f, Point2f> > MatchBF(BRIEF::Features& f1, BRIEF::Features& f2)
{
	vector< pair<Point2f, Point2f> > result;
	if (f2.size() > 0)
	{
#pragma omp parallel reduction(merge: result)
		for (int i = 0; i < f1.size(); ++i)
		{
			unsigned int min = INT_MAX;
			BRIEF::Feature f;
			Point2f pos;
			BRIEF::Feature p = f1[i];
			for (int j = 0; j < f2.size(); ++j)
			{
				unsigned int distance = f1[i] - f2[j];
				unsigned int L2 = pow(p.position.x - f2[j].position.x, 2) + pow(p.position.y - f2[j].position.y, 2);
				if (distance < 30 && L2<10000)
				{
					distance *= L2;
					if (distance < min)
					{
						min = distance;
						pos = f2[j].position;
					}
				}
			}
			if (min < INT_MAX)
			{
				result.push_back(pair<Point2f, Point2f>(pos, f1[i].position));
			}
		}
	}
	return result;
}
