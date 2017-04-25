#include "brief.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
namespace BRIEF
{
	namespace matcher
	{

	
		vector< pair<Point2f, Point2f> > MatchBF(BRIEF::Features& f1, BRIEF::Features& f2, const int threshold)
		{
			vector< pair<Point2f, Point2f> > result;
			vector< int > rs(f1.size());
			if (f2.size() > 0)
			{
	#pragma omp parallel for
				for (int i = 0; i < f1.size(); ++i)
				{
					unsigned int min = INT_MAX;
					int minj = 0;
					BRIEF::Feature& f = BRIEF::Feature();
					for (int j = 0; j < f2.size(); ++j)
					{
						unsigned int distance = f1[i] - f2[j];
						if (distance < threshold)
						{
							if (distance < min)
							{
								min = distance;
								minj = j;
							}
						}
					}
					if (min < INT_MAX)
					{
						rs[i] = minj;
					}
				}
				for (int i = 0; i < rs.size(); ++i)
				{
					int j = rs[i];
					if (rs[i] > 0)
					{
						result.push_back(pair<Point2f, Point2f>(f2[j].position, f1[i].position));
					}
				}
			}
			return result;
		}
	}
}