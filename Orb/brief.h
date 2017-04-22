#pragma once
#ifndef BRIEF_H
#define BRIEF_H
#include <opencv2\core.hpp>
#include <vector>

#define BRIEF_DEFAULT_WINDOW_SIZE 31
#define BRIEF_DEFAULT_TEST_COUNT 256
#define sBRIEF_DEFAULT_LUT_SIZE 30
class BRIEF
{

public:
	class Feature;
	class Features : public std::vector<Feature>
	{
	public:
		Features() : vector<Feature>()
		{

		}
		Features(int length) : vector<Feature>(length)
		{

		}
	};
	BRIEF();
	BRIEF(int S);
	virtual Features extractFeature(uint8_t* image, std::vector<cv::Point2d>& positions, const int width, const int height) const;
	virtual Features extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, const int width, const int height) const;
	unsigned int DistanceBetween(Feature& f1, Feature& f2) const;

	class Feature
	{
	public:
		Feature();
		
		cv::Point2d position;
		int operator -(Feature&) const;
		friend std::ostream& operator<<(std::ostream& os, const Feature& dt);
		inline void setbit(int pos, bool v);
		union {
			//8*uint32 = 8*32bit which is 256 binary tests
			unsigned int value[8];
			__m256 f_vect;
		};
	};

protected:
	int size;
	virtual void GenerateBinaryTests(int* x, int* y, const int count, const int dim);
private:
	//need 512 coordinates to store 256 binary tests
	int xp[512];
	int yp[512];
};

class rBRIEF : public BRIEF
{
private:
	std::vector< std::vector<int> > lutx;
	std::vector< std::vector<int> > luty;
protected:
	int angleCount;
	virtual void generateLUT(const int count, const int dim, const int _angleCount,int* _xp,int* _yp);

public:

	rBRIEF();
	rBRIEF(int S);
	rBRIEF(int S, int count);
	rBRIEF(int S, int* _xp, int* _yp);
	~rBRIEF();
	//indexer to obtain underlying BRIEF test pattern
	std::pair< std::vector<int>, std::vector<int>> operator [](int i) const;
	virtual Features extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, const int width, const int height) const;
	virtual Features extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, std::vector<float>& angles, const int width, const int height) const;
};



std::vector< std::pair<cv::Point2f, cv::Point2f> > MatchBF(BRIEF::Features& f1, BRIEF::Features& f2, int threshold = 30);


//split all features into 256 bins
class LSHashSet
{
public:
	LSHashSet();
	LSHashSet(uint8_t bitmask);
	BRIEF::Features& operator[](BRIEF::Feature& f);
	void InsertRange(BRIEF::Features& features);

protected:
	uint8_t hash(BRIEF::Feature& feature);
	uint8_t bitmask;
	BRIEF::Features table[256];
};


class MultiLSHashTable
{
public:
	MultiLSHashTable()
	{
		for (int i = 0; i < 8; ++i)
		{
			hs.push_back(LSHashSet(i));
		}
	}
	void InsertRange(BRIEF::Features& features);
	
	std::vector< std::pair<cv::Point2f, cv::Point2f> > Hash_Match(BRIEF::Features& fs, const int max_distance = 30);
protected:
	std::pair<int, BRIEF::Feature> Hash_Find(BRIEF::Feature& f, const int max_distance = 30);
private:
	std::vector<LSHashSet> hs;
};




#endif