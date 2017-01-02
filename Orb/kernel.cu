#include <windows.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <arrayfire.h>
#include "Profiler.cuh"
#include "Memory.cuh"
#include "FAST.cuh"
#include "Utility.cuh"
#include "brief.h"
#include <math.h>
using namespace cv;
using namespace af;
using namespace std;


#include <sstream>
#include <fstream>

#define Harris_Threshold 50000000
#define FAST_Corner_Limit 500000


struct FeatureVector
{
	unsigned int x, y, value[8];
};

//blockdim = 32x32, load 32 features from both segments
__global__ void cuda_BRIEF_match_bruteforce(unsigned int* resultMap, FeatureVector *f1, FeatureVector *f2,int c1,int c2)
{
	int y = threadIdx.y + blockIdx.y*gridDim.y;
	int x = threadIdx.x + blockIdx.x*gridDim.x;
	int index = y*2000 + x;
	__shared__ unsigned int features1[32][10];
	__shared__ unsigned int features2[32][10];
	if (y < c2 && x < c1)
	{
		if (threadIdx.y < 8)
		{
			features1[threadIdx.x][threadIdx.y] = f1[x].value[threadIdx.y];
		}
		else if (threadIdx.y >= 8 && threadIdx.y < 16)
		{
			features2[threadIdx.x][threadIdx.y - 8] = f2[x].value[threadIdx.y - 8];
		}

		__syncthreads();
		int distance = 0;
		for (int i = 0; i < 8; ++i)
		{
			distance += __popc(features1[threadIdx.x][i] ^ features2[threadIdx.y][i]);
		}
		resultMap[index] = distance < 50 ? distance : 255;
	}
	else
		resultMap[index] = 255;
}



int main(int argc,char** argv)
{
	static unsigned int resultMapHost[2000 * 2000];
	BRIEF extractor(31);
	Profiler profiler;
	VideoCapture cap; 	

	
	//cap.open("I:\\Downloads\\Act of Valor (2012) [1080p]\\Act.of.Valor.2012.1080p.BrRip.x264.YIFY.mp4");
	cap.open("C:\\Users\\timya\\Desktop\\484286440.mp4");
	if (!cap.isOpened())
		return -1;
	Mat frame;
	cap.read(frame);
	Mat grey = Mat(frame.rows, frame.cols, CV_8UC1);
	Mat out = Mat(frame.rows, frame.cols, CV_8UC1);
	cuArray<uchar> buf1(frame.cols*frame.rows);
	cuArray<uchar> bufx(frame.cols*frame.rows);
	cuArray<uint4> AngleMap(FAST_Corner_Limit);
	cuArray<FeatureVector> gpuFeatures0(2000);
	cuArray<FeatureVector> gpuFeatures1(2000);
	cuArray<unsigned int> resultMap(2000 * 2000);



	namedWindow("output");
	BRIEF::Features features_old;
	for (int fc=0;;++fc)
	{
		if (!cap.read(frame))break;
		int points = 0;
		cvtColor(frame, grey, CV_BGR2GRAY);	
		//medianBlur(grey, out, 3);

		buf1.upload(grey.data);
		profiler.Start();
		{
			FAST <<< dim3(frame.cols / FAST_TILE, frame.rows / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >>> (buf1, bufx,20, grey.cols, grey.rows);
		}

		std::stringstream ss;
		ss << "frame: " << profiler.Count() << " | " << profiler.Stop() <<"us | ";		
		bufx.download(out.data);

		vector<uint4> corners;
		for (uint i = 0; i < grey.cols; ++i)
			for (uint j = 0; j < grey.rows; ++j)
			{
				uint cvalue = out.data[i + j*grey.cols];
				if (cvalue > 0)
				{
					corners.push_back(uint4{ i,j,0,0});
					points++;
				}
			}
		int processed_count = points > FAST_Corner_Limit ? FAST_Corner_Limit : points;
		AngleMap.upload(corners.data(), processed_count);
		FAST_Refine << < corners.size() / 32, 32 >> > (buf1, AngleMap, corners.size(), grey.cols, grey.rows);
		AngleMap.download(corners.data(), processed_count);	
		std::sort(corners.begin(), corners.end(), [](const uint4&x, const uint4& y) {
			return x.w > y.w;
		});

		vector<Point2d> poi;
		for (int i = 0; i < min(processed_count,2000); ++i)
		{			
			if (i < 2000 )
			{
				poi.push_back(Point2d(Point2d(corners[i].x, corners[i].y)));
				
			}	
			else
			{
				ss << i << " Corners";
				break;
			}		
		}
		BRIEF::Features features;
		extractor.extractFeature(grey.data, poi, features, grey.cols, grey.rows);
		


		
		
		/*{
			vector<FeatureVector> f1;
			int  vc = features.size() > 2000 ? 2000 : features.size();
			for (int i = 0; i < vc; ++i)
			{
				FeatureVector fv;
				fv.x = features[i].position.x;
				fv.y = features[i].position.y;
				memcpy(fv.value, features[i].value, 8 * sizeof(unsigned int));
				f1.push_back(fv);
			}
			int vc_old = features_old.size() > 2000 ? 2000 : features_old.size();
			if (fc % 2)
			{
				gpuFeatures0.upload(f1.data(),vc);
				if (fc > 0)
				cuda_BRIEF_match_bruteforce<<<dim3(vc/32, vc_old/32), dim3(32,32)>>>(resultMap, gpuFeatures0, gpuFeatures1,vc,vc_old);
			}
			else
			{
				gpuFeatures1.upload(f1.data(), vc);
				if (fc > 0)
				cuda_BRIEF_match_bruteforce << <dim3(vc / 32, vc_old / 32), dim3(32, 32) >> >(resultMap, gpuFeatures1, gpuFeatures0, vc, vc_old);
			}
			resultMap.download(resultMapHost);
			cout << resultMapHost[1]<<endl;
			for (int i = 0; i < vc; ++i)
			{

				int min = INT_MAX;
				int minj = -1;
				for (int j = 0; j < vc_old; ++j)
				{
					
					if (min > resultMapHost[j * 2000 + i])
					{
						min = resultMapHost[j * 2000 + i];
						minj = j;
					}
				}
				if(minj>=0)
				line(frame, features[i].position, features_old[minj].position, Scalar(255, 255, 0), 1, cv::LineTypes::LINE_AA);
			}
		}*/


		if (features_old.size() > 0)
		{
			#pragma omp parallel 
			for (int i = 0; i < features.size(); ++i)
			{
				unsigned int min = INT_MAX;
				BRIEF::Feature f;
				Point2d pos;
				BRIEF::Feature p = features[i];
				for (int j = 0; j < features_old.size(); ++j)
				{
					
						unsigned int distance = features[i] - features_old[j];
						unsigned int L2 = pow(p.position.x - features_old[j].position.x, 2) + pow(p.position.y - features_old[j].position.y, 2);
						if (distance < 50 && L2<2500)
						{
							distance *= L2;
							if (distance < min)
							{
								min = distance;
								pos = features_old[j].position;
							}
						}
				}
				if (min < INT_MAX)
				{
					line(frame, features[i].position, pos, Scalar(255, 255, 0), 1, cv::LineTypes::LINE_AA);
					circle(frame, pos, 2, Scalar(0, 255,0));
			    }
					
				else
				{
					circle(frame, Point2d(p.position), 2, Scalar(0, 0, 255));
				}
			}
		}
			
		features_old = features;
		cv::putText(frame, ss.str(), Point2d(grey.cols / 2 - 500, grey.rows / 6 * 5), HersheyFonts::FONT_HERSHEY_DUPLEX, 2, Scalar(255,255, 255, 255));
	  	cv::imshow("output", frame);
		
		 if (waitKey(1) >= 0) break;
	}
	return 0;
}