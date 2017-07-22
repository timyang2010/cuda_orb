#include "BRIEF.h"
using namespace std;
using namespace cv;
namespace ty
{

	LSHashSet::LSHashSet()
	{
		bitmask = 0;
	}
	LSHashSet::LSHashSet(unsigned char _bitmask)
	{
		bitmask = _bitmask;
	}
	unsigned char LSHashSet::hash(BRIEF::Feature& feature)
	{
		return feature.value[bitmask];
	}

	std::vector<BRIEF::Feature>& LSHashSet::operator[](BRIEF::Feature& f)
	{
		return table[hash(f)];
	}

	void LSHashSet::InsertRange(std::vector<BRIEF::Feature>& features)
	{
		for (std::vector<BRIEF::Feature>::iterator it = features.begin(); it != features.end(); ++it)
		{
			table[hash(*it)].push_back(*it);
		}
	}
	void MultiLSHashTable::InsertRange(std::vector<BRIEF::Feature>& features)
	{
		#pragma omp parallel for
		for (int i = 0; i < BRIEF_DEFAULT_WORDLENGTH; ++i)
		{
			hs[i].InsertRange(features);
		}
	}

#include <algorithm>
	std::pair<int, BRIEF::Feature> MultiLSHashTable::Hash_Find(BRIEF::Feature& f, const int max_distance)
	{
		std::vector<std::pair<int, BRIEF::Feature>> distances(BRIEF_DEFAULT_WORDLENGTH);
#pragma omp parallel for
		for (int i = 0; i < BRIEF_DEFAULT_WORDLENGTH; ++i)
		{
			std::vector<BRIEF::Feature> ff = hs[i][f];
			int min = INT_MAX;
			BRIEF::Feature minf;
			for (auto t : ff)
			{
				int distance = t - f;
				if (min > distance)
				{
					min = distance;
					minf = t;
				}
			}
			distances[i] = std::pair<int, BRIEF::Feature>(min, minf);
		}

		int min = INT_MAX;
		std::pair<int, BRIEF::Feature> minf;
		for (auto t : distances)
		{
			if (min > t.first)
			{
				minf = t;
			}
		}
		return minf;
	}

	std::vector< std::pair<cv::Point2f, cv::Point2f> > MultiLSHashTable::hashMatch(std::vector<BRIEF::Feature>& fs, const int max_distance)
	{
		std::vector< std::pair<cv::Point2f, cv::Point2f> > mpairs;
		for (auto t : fs)
		{
			auto p = Hash_Find(t, max_distance);
			if (p.first < max_distance)
			{
				mpairs.push_back(std::pair<cv::Point2f, cv::Point2f>(p.second.position, t.position));
			}
		}
		return mpairs;
	}

}

namespace ty
{

	vector< pair<Point2f, Point2f> > BRIEF::matchFeatures(std::vector<BRIEF::Feature>& f1, std::vector<BRIEF::Feature>& f2, const int threshold)
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
				else
				{
					rs[i] = -1;
				}
			}
			for (int i = 0; i < rs.size(); ++i)
			{
				int j = rs[i];
				if (j >= 0)
				{
					result.push_back(pair<Point2f, Point2f>(f1[i].position, f2[j].position));
				}
			}
		}
		return result;
	}

}