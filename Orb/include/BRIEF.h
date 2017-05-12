#pragma once
#ifndef BRIEF_H
#define BRIEF_H
#include <opencv2/core.hpp>
#include <vector>
#include <immintrin.h>
namespace ty
{
#define BRIEF_DEFAULT_WORDLENGTH 8
#define BRIEF_DEFAULT_WINDOW_SIZE 31
#define BRIEF_DEFAULT_SUBWINDOW_SIZE 5
#define BRIEF_DEFAULT_TEST_COUNT 256
#define sBRIEF_DEFAULT_LUT_SIZE 30

	struct Keypoint
	{
		Keypoint()
		{
			x = 0;
			y = 0;
			z = 0;
			w = 0;
		}
		Keypoint(float _x,float _y)
		{
			x = _x;
			y = _y;
		}
		float x, y, z, w;
		cv::Point2f position()
		{
			return cv::Point2f(x, y);
		}
	};

	class BRIEF
	{

	public:
		class Feature;
		struct BinaryTest;

		BRIEF();
		BRIEF(int S);
		BRIEF(std::vector<BinaryTest>& ts);
		virtual std::vector<Feature> extractFeatures(uint8_t** image, std::vector<Keypoint>& positions) const;
		unsigned int DistanceBetween(Feature& f1, Feature& f2) const;

		class Feature
		{
		public:
			Feature();

			cv::Point2f position;
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
			signed char x1;
			signed char y1;
			signed char x2;
			signed char y2;
			BinaryTest Rotate(double _cos, double _sin)  const;
			friend std::ostream& operator<<(std::ostream& os, const BinaryTest& dt);
		};

	protected:
		std::vector<BRIEF::BinaryTest> GenerateBinaryTests(const int count, const int dim);
	private:
		std::vector<BRIEF::BinaryTest> _tests;
	};

	class rBRIEF : public BRIEF
	{
	private:
		std::vector< std::vector<BRIEF::BinaryTest> > lut;
	protected:
		int angleCount;
		virtual void generateLUT(std::vector<BRIEF::BinaryTest> tests, const int _angleCount);

	public:

		rBRIEF();
		rBRIEF(int S);
		rBRIEF(int S, int count);
		rBRIEF(std::vector<BinaryTest>& ts);
		~rBRIEF();
		//indexer to obtain underlying BRIEF test pattern
		std::vector<BRIEF::BinaryTest> operator [](int i) const;
		virtual std::vector<Feature> extractFeatures(uint8_t** image, std::vector<Keypoint>& positions) const;
	};


	std::vector< std::pair<cv::Point2f, cv::Point2f> > MatchBF(std::vector<BRIEF::Feature>& f1, std::vector<BRIEF::Feature>& f2, int threshold = 30);


	//split all features into 256 bins
	class LSHashSet
	{
	public:
		LSHashSet();
		LSHashSet(unsigned char bitmask);
		std::vector<BRIEF::Feature>& operator[](BRIEF::Feature& f);
		void InsertRange(std::vector<BRIEF::Feature>& features);

	protected:
		unsigned char hash(BRIEF::Feature& feature);
		unsigned char bitmask;
		std::vector<BRIEF::Feature> table[256];
	};


	class MultiLSHashTable
	{
	public:
		MultiLSHashTable()
		{
			for (int i = 0; i < BRIEF_DEFAULT_WORDLENGTH; ++i)
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
		void extractFeatures(uint8_t** image, std::vector<Keypoint>& positions);

		//returns a set of optimized BRIEF tests based on input keypoints
		std::vector<BRIEF::BinaryTest> Optimize(float stp = 0.3f,float delta = 0.03f);

		//utility function, compute variance of given BRIEF::Feature set
		void generateTests(
			int windowSize = BRIEF_DEFAULT_WINDOW_SIZE, 
			int subWindowSize = BRIEF_DEFAULT_SUBWINDOW_SIZE, 
			int min_distance = 4,
			int scale = 1
		);

		std::vector<candidate> candidates;
	protected:
		//encapsulate metadata for a single binary test and its test results	
		class candidate
		{
		public:
			candidate(BRIEF::BinaryTest _test);
			double mean();
			double stddev();
			std::vector<BRIEF::BinaryTest> tests;
			std::vector<unsigned short> testResult;
			void computeRank();
			double rank;
		private:
			double _mean = -1;
			double _stddev = -1;
		};
			
		bool checkCorrelation(candidate& c1, std::vector<candidate>& c2,double thres);
		//compute absolute correlation between to binary tests	
		double correlation(candidate& c1, candidate& c2);
		
	private:
		
	};

}


#endif