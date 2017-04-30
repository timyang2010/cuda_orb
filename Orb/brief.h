#pragma once
#ifndef BRIEF_H
#define BRIEF_H
#include <opencv2/core.hpp>
#include <vector>
#include <immintrin.h>
namespace BRIEF
{

#define BRIEF_DEFAULT_WINDOW_SIZE 31
#define BRIEF_DEFAULT_TEST_COUNT 256
#define sBRIEF_DEFAULT_LUT_SIZE 30
	class BRIEF
	{

	public:
		class Feature;
		struct BinaryTest;

		BRIEF();
		BRIEF(int S);
		BRIEF(std::vector<BinaryTest> ts);
		virtual std::vector<Feature> extractFeatures(uint8_t** image, std::vector<cv::Point2d>& positions) const;
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

		struct BinaryTest
		{
			int8_t x1;
			int8_t y1;
			int8_t x2;
			int8_t y2;
			BinaryTest Rotate(double _cos, double _sin)  const;
		};

	protected:
		int size;
		std::vector<BRIEF::BinaryTest> GenerateBinaryTests(const int count, const int dim);
	private:
		std::vector<BRIEF::BinaryTest> _tests;
	};

	class rBRIEF : public BRIEF
	{
	private:
		std::vector< std::vector<BRIEF::BinaryTest> > lut;
		//std::vector< std::vector<int> > luty;
	protected:
		int angleCount;
		virtual void generateLUT(std::vector<BRIEF::BinaryTest> tests, const int _angleCount);

	public:

		rBRIEF();
		rBRIEF(int S);
		rBRIEF(int S, int count);
		~rBRIEF();
		//indexer to obtain underlying BRIEF test pattern
		std::vector<BRIEF::BinaryTest> operator [](int i) const;
		virtual std::vector<Feature> extractFeatures(uint8_t** image, std::vector<cv::Point2d>& positions) const;
		virtual std::vector<Feature> extractFeatures(uint8_t** image, std::vector<cv::Point2d>& positions, std::vector<float>& angles) const;
	};


	std::vector< std::pair<cv::Point2f, cv::Point2f> > MatchBF(std::vector<BRIEF::Feature>& f1, std::vector<BRIEF::Feature>& f2, int threshold = 30);


	//split all features into 256 bins
	class LSHashSet
	{
	public:
		LSHashSet();
		LSHashSet(uint8_t bitmask);
		std::vector<BRIEF::Feature>& operator[](BRIEF::Feature& f);
		void InsertRange(std::vector<BRIEF::Feature>& features);

	protected:
		uint8_t hash(BRIEF::Feature& feature);
		uint8_t bitmask;
		std::vector<BRIEF::Feature> table[256];
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
		void InsertRange(std::vector<BRIEF::Feature>& features);
		std::vector< std::pair<cv::Point2f, cv::Point2f> > Hash_Match(std::vector<BRIEF::Feature>& fs, const int max_distance = 30);
	protected:
		std::pair<int, BRIEF::Feature> Hash_Find(BRIEF::Feature& f, const int max_distance = 30);
	private:
		std::vector<LSHashSet> hs;
	};


	//encapsulate training and searching methods for rBRIEF test patterns
	class Optimizer
	{
	public:
		class candidate;
		Optimizer();
		void extractFeatures(uint8_t** image, std::vector<cv::Point2f>& positions);
		void extractFeatures(uint8_t** image, std::vector<cv::Point2f>& positions, std::vector<float>& angles);

		//returns a set of optimized BRIEF tests based on input keypoints
		std::vector<BRIEF::BinaryTest> Optimize(int length = BRIEF_DEFAULT_TEST_COUNT);

		//utility function, compute variance of given BRIEF::Feature set
		double computeVariance(std::vector<BRIEF::Feature>& features);
		class candidate
		{
		public:
			candidate(BRIEF::BinaryTest _test)
			{
				test = _test;
			}
			BRIEF::BinaryTest test;
			std::vector<unsigned short> testResult;
			void computeRank();
			double rank;
		};
		void generateTests(int windowSize = BRIEF_DEFAULT_WINDOW_SIZE, int subWindowSize = 5);
		std::vector<candidate> candidates;
	protected:
		//encapsulate metadata for a single binary test and its test results
		
		
		double correlation(candidate& c1, candidate& c2);
		//sort candidates by their distance to mean
		//compute absolute correlation between to binary tests			
	private:
		
	};

}


#endif