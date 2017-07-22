#include "BRIEF.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#define BLOCKDIM 16
#define MAX_FEATURE_CNT 1000
#define FEATURE_LENGTH 8
using namespace ty;
__global__ void bf_hamming_dist(unsigned int* d_map, unsigned int* xfeatures, unsigned int* yfeatures)
{
	int gx = threadIdx.x + blockIdx.x*blockDim.x;
	int gy = threadIdx.y + blockIdx.y*blockDim.y;
	int g = (MAX_FEATURE_CNT*gy + gx);
	__shared__ unsigned int x_features[BLOCKDIM][BLOCKDIM];
	__shared__ unsigned int y_features[BLOCKDIM][BLOCKDIM];
	//memory load with first 16 rows in the block
	if (threadIdx.y < FEATURE_LENGTH)
	{
		x_features[threadIdx.x][threadIdx.y] = xfeatures[gx*FEATURE_LENGTH + threadIdx.y];
	}
	else if (threadIdx.y >= FEATURE_LENGTH)
	{
		y_features[threadIdx.x][threadIdx.y - FEATURE_LENGTH] = yfeatures[gy*FEATURE_LENGTH + threadIdx.y - FEATURE_LENGTH];
	}
	__syncthreads();
	int dist = 0;
	//compute bit difference by xor and CUDA intrinsics
	for (int i = 0; i < FEATURE_LENGTH; ++i)
	{
		dist += __popc(x_features[threadIdx.x][i] ^ y_features[threadIdx.y][i]);
	}
	d_map[g] = dist;
}

//apply merge sort on rows
__global__ void reduce_argmax(unsigned int* d_map)
{

}
std::vector< std::pair<cv::Point2f, cv::Point2f> > BRIEF::matchFeatures_gpu(std::vector<BRIEF::Feature>& f1, std::vector<BRIEF::Feature>& f2, int threshold)
{
	static unsigned int *dmap, *x, *y;
	static bool init = false;
	static unsigned int *dmap_host;
	if (!init)
	{
		cudaMalloc(&dmap, MAX_FEATURE_CNT*MAX_FEATURE_CNT);
		cudaMalloc(&x, MAX_FEATURE_CNT*FEATURE_LENGTH);
		cudaMalloc(&y, MAX_FEATURE_CNT*FEATURE_LENGTH);
		cudaMemset(x, 0, MAX_FEATURE_CNT*FEATURE_LENGTH);
		cudaMemset(x, 0, MAX_FEATURE_CNT*FEATURE_LENGTH);
		dmap_host = new unsigned int[MAX_FEATURE_CNT*MAX_FEATURE_CNT];
		init = true;
	}
	std::vector<unsigned int> fx, fy;

	for (BRIEF::Feature& f : f1)
		for(int i=0;i<8;++i)
			fx.push_back(f.value[i]);
	for (BRIEF::Feature& f : f2)
		for (int i = 0; i<8; ++i)
			fy.push_back(f.value[i]);

	cudaMemcpy(x, fx.data(), fx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(y, fy.data(), fy.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
	bf_hamming_dist << <dim3(MAX_FEATURE_CNT / BLOCKDIM, MAX_FEATURE_CNT / BLOCKDIM), dim3(BLOCKDIM, BLOCKDIM) >> > (dmap, x, y);
	cudaMemcpy(dmap_host, dmap, MAX_FEATURE_CNT*MAX_FEATURE_CNT * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	std::vector< std::pair<cv::Point2f, cv::Point2f> > m;
	int min = INT_MAX;
	int mini = 0;
	for (int row = 0; row < f2.size()*MAX_FEATURE_CNT; row += MAX_FEATURE_CNT)
	{
		for (int i = 0; i < f1.size(); ++i)
		{
			if (min > dmap_host[row + i])
			{
				min = dmap_host[row + i];
				mini = i;
			}
		}
		if(min<threshold)
			m.push_back(std::pair<cv::Point2f, cv::Point2f>(f1[mini].position, f2[row/ MAX_FEATURE_CNT].position));
		min = INT_MAX;
	}

	return m;
}

