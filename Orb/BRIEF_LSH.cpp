#include "brief.h"

LSHashSet::LSHashSet()
{
	bitmask = 0;
}
LSHashSet::LSHashSet(uint8_t _bitmask)
{
	bitmask = _bitmask;
}
uint8_t LSHashSet::hash(BRIEF::Feature& feature)
{
	return feature.value[bitmask];
}

BRIEF::Features& LSHashSet::operator[](BRIEF::Feature& f)
{
	return table[hash(f)];
}

void LSHashSet::InsertRange(BRIEF::Features& features)
{
	for (BRIEF::Features::iterator it = features.begin(); it != features.end(); ++it)
	{
		table[hash(*it)].push_back(*it);
	}
}
void MultiLSHashTable::InsertRange(BRIEF::Features& features)
{
#pragma omp parallel for
	for (int i = 0; i < 8; ++i)
	{
		hs[i].InsertRange(features);
	}
}

#include <algorithm>
std::pair<int, BRIEF::Feature> MultiLSHashTable::Hash_Find(BRIEF::Feature& f,const int max_distance)
{
	std::vector<std::pair<int,BRIEF::Feature>> distances(8);
#pragma omp parallel for
	for (int i = 0; i < 8; ++i)
	{
		BRIEF::Features ff = hs[i][f];
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

std::vector< std::pair<cv::Point2f, cv::Point2f> > MultiLSHashTable::Hash_Match(BRIEF::Features& fs, const int max_distance)
{
	std::vector< std::pair<cv::Point2f, cv::Point2f> > mpairs;
	for (auto t : fs)
	{
		auto p = Hash_Find(t, max_distance);
		if (p.first < max_distance)
		{
			mpairs.push_back(std::pair<cv::Point2f, cv::Point2f>(p.second.position,t.position));
		}
	}
	return mpairs;
}
