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

		struct BinaryTest
		{
			int8_t x1;
			int8_t y1;
			int8_t x2;
			int8_t y2;
			BinaryTest Rotate(double _cos, double _sin)  const;
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
		virtual Features extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, const int width, const int height) const;
		virtual Features extractFeature(uint8_t** image, std::vector<cv::Point2d>& positions, std::vector<float>& angles, const int width, const int height) const;
	};


	namespace matcher
	{

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

	}

	namespace train
	{
		//encapsulate training and searching methods for rBRIEF test patterns
		class rBRIEF_pattern_trainer
		{
		public:
			rBRIEF_pattern_trainer();
			void run_batch_keypoints(uint8_t** image, std::vector<cv::Point2d>& positions, const int width, const int height);
			void run_batch_keypoints(uint8_t** image, std::vector<cv::Point2d>& positions, std::vector<float>& angles, const int width, const int height);
			std::vector<BRIEF::BinaryTest> generatePattern();
		protected:
			//encapsulate metadata for a single binary test and its test results
			class candidate
			{
			public:
				BRIEF::BinaryTest test;
				std::vector<bool> testResult;
				int rank;
			};
			void generateCompleteTestSpace(int windowSize = BRIEF_DEFAULT_WINDOW_SIZE,int subWindowSize = 5);

			//sort candidates by their distance to mean
			void RankByDistance(double mean = 0.5f);

			//compute absolute correlation between to binary tests
			double computeCorrelation(candidate& c1, candidate& c2);
		private:
			std::vector<candidate> candidates;
		};
	}

}


#endif