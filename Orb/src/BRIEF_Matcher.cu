#include "BRIEF.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#define BLOCKDIM 16
#define MAX_FEATURE_CNT 1024
#define FEATURE_LENGTH 8
using namespace ty;
__global__ void bf_hamming_dist(unsigned char * d_map, unsigned int* xfeatures, unsigned int* yfeatures)
{
	int gx = threadIdx.x + blockIdx.x*BLOCKDIM;
	int gy = threadIdx.y + blockIdx.y*BLOCKDIM;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	//int g= blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
	int g = (MAX_FEATURE_CNT*gy + gx);
	__shared__ unsigned int x_features[BLOCKDIM][FEATURE_LENGTH];
	__shared__ unsigned int y_features[BLOCKDIM][FEATURE_LENGTH];
	//memory load with first 16 rows in the block
	if (threadIdx.y < FEATURE_LENGTH)
	{
		x_features[threadIdx.x][threadIdx.y] = xfeatures[gx*FEATURE_LENGTH + threadIdx.y];
	}
	if (threadIdx.x < FEATURE_LENGTH)
	{
		y_features[threadIdx.y][threadIdx.x] = yfeatures[gy*FEATURE_LENGTH + threadIdx.x];
	}
	__syncthreads();
	int dist = 0;
	//compute bit difference by xor and CUDA intrinsics
	#pragma unroll
	for (int i = 0; i < FEATURE_LENGTH; ++i)
	{
		dist += __popc(x_features[threadIdx.x][i] ^ y_features[threadIdx.y][i]);
	}
	d_map[g] = dist;
}

//apply merge sort on rows
__global__ void reduce_argmin(unsigned char* d_map,unsigned int* d_min,int max_x, int threshold)
{
	//1024 1-d blocks, 32 thread per block
	unsigned char *arr = d_map + blockIdx.x*MAX_FEATURE_CNT;
	__shared__ unsigned char buffer[1024];
	__shared__ unsigned int min_dist[32];
	__shared__ unsigned int min_idx[32];

	min_dist[threadIdx.x] = 255;
	for (int i = 0; i < MAX_FEATURE_CNT; i+=32)
	{
		buffer[threadIdx.x + i] = arr[threadIdx.x + i];
	}
	__syncthreads();
	for (int i = 0; i < 32; ++i)
	{
		int idx = threadIdx.x * 32 + i;
		if (idx > max_x)break;
		unsigned char v = buffer[idx];
		if (min_dist[threadIdx.x] > v)
		{
			min_dist[threadIdx.x] = v;
			min_idx[threadIdx.x] = idx;
		}
	}
	__syncthreads();
	if (threadIdx.x == 0)
	{
		unsigned int local_min_dist = 255;
		unsigned int local_min_idx = 0;
		for (int i = 0; i < 32; ++i)
		{
			if (local_min_dist > min_dist[i])
			{
				local_min_dist = min_dist[i];
				local_min_idx = min_idx[i];
			}
		}
		if (local_min_dist < threshold)
			d_min[blockIdx.x] = local_min_idx;
		else
			d_min[blockIdx.x] = -1;
	}
}
#include <opencv2\highgui.hpp>
#include <cuda_runtime.h>
std::vector< std::pair<cv::Point2f, cv::Point2f> > BRIEF::matchFeatures_gpu(std::vector<BRIEF::Feature>& f1, std::vector<BRIEF::Feature>& f2, int threshold)
{
	static cv::Mat rm(MAX_FEATURE_CNT, MAX_FEATURE_CNT, CV_8UC1);
	static unsigned int *x, *y, *x_host, *y_host;
	static unsigned char *dmap;
	static unsigned char *dmap_host;
	static unsigned int *dmin,*dmin_host;
	if (!dmap)
	{
		cudaMalloc(&dmap, MAX_FEATURE_CNT*MAX_FEATURE_CNT * sizeof(unsigned char));
		dmap_host = new unsigned char[MAX_FEATURE_CNT*MAX_FEATURE_CNT];
		x_host = new unsigned int[MAX_FEATURE_CNT*FEATURE_LENGTH];
		y_host = new unsigned int[MAX_FEATURE_CNT*FEATURE_LENGTH];
		dmin_host = new unsigned int[MAX_FEATURE_CNT];
		cudaMalloc(&x, MAX_FEATURE_CNT*FEATURE_LENGTH * sizeof(unsigned int));
		cudaMalloc(&y, MAX_FEATURE_CNT*FEATURE_LENGTH * sizeof(unsigned int));
		cudaMalloc(&dmin, MAX_FEATURE_CNT * sizeof(unsigned int));
	}
	std::vector< std::pair<cv::Point2f, cv::Point2f> > m(f2.size());
	if (f1.size() == 0 || f2.size() == 0)return m;
	for (int j = 0; j < f1.size(); ++j)
		for (int i = 0; i < 8; ++i)
			x_host[j*FEATURE_LENGTH + i] = f1[j].value[i];
	for (int j = 0; j < f2.size(); ++j)
		for (int i = 0; i < 8; ++i)
			y_host[j*FEATURE_LENGTH + i] = f2[j].value[i];

	cudaMemcpy(x, x_host, MAX_FEATURE_CNT * sizeof(unsigned int)*FEATURE_LENGTH, cudaMemcpyHostToDevice);
	cudaMemcpy(y, y_host, MAX_FEATURE_CNT * sizeof(unsigned int)*FEATURE_LENGTH, cudaMemcpyHostToDevice);

	bf_hamming_dist << <dim3(MAX_FEATURE_CNT / BLOCKDIM, MAX_FEATURE_CNT / BLOCKDIM), dim3(BLOCKDIM, BLOCKDIM) >> > (dmap, x, y);
	reduce_argmin << <dim3(f2.size()), dim3(32) >> > (dmap, dmin, f1.size(), threshold);
	cudaMemcpy(dmin_host, dmin, MAX_FEATURE_CNT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	/*cudaMemcpy(rm.data, dmap, MAX_FEATURE_CNT*MAX_FEATURE_CNT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cv::imshow("rn", rm);*/
	#pragma omp parallel for
	for (int j = 0; j < f2.size(); ++j)
	{
		if(dmin_host[j]!=-1)
			m[j] = std::pair<cv::Point2f, cv::Point2f>(f1[dmin_host[j]].position, f2[j].position);
	}

	return m;
}

