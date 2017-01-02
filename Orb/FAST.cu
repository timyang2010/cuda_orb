
#define TILE_RADIUS FAST_TILE/2
#include "FAST.cuh"
#include "device_launch_parameters.h"
__global__ void FAST(unsigned char* __restrict__ inputImage, unsigned char* __restrict__ cornerMap, const int threshold, const int width, const  int height)
{
	const int offsetX[27] = { 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2 };
	const int offsetY[27] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2 };
	__shared__ int tile[FAST_TILE * 2 + 1][FAST_TILE * 2 + 1];
	int hblocks = width / FAST_TILE;
	int vblocks = height / FAST_TILE;
	int sourceX = blockIdx.x*blockDim.x + threadIdx.x;
	int sourceY = blockIdx.y*blockDim.y + threadIdx.y;
	int source = sourceX + sourceY*width;

	if (blockIdx.x > 0 && blockIdx.x < hblocks  && blockIdx.y>0 && blockIdx.y < vblocks)
	{
		for (int i = 0; i <= FAST_TILE; i += FAST_TILE)
			for (int j = 0; j <= FAST_TILE; j += FAST_TILE)
			{
				int xdestX = threadIdx.x + i;
				int xdestY = threadIdx.y + j;
				int xsourceX = blockIdx.x*blockDim.x + xdestX - TILE_RADIUS;
				int xsourceY = blockIdx.y*blockDim.y + xdestY - TILE_RADIUS;
				int xsource = xsourceX + xsourceY*width;
				tile[xdestX][xdestY] = inputImage[xsource];
			}
		__syncthreads();
		//FAST Algorithm
		int highCount = 0, lowCount = 0;
		int cX = threadIdx.x + TILE_RADIUS, cY = threadIdx.y + TILE_RADIUS;
		int center = tile[cX][cY];
		int t_low = (center < threshold) ? 0 : center - threshold;
		int t_high = (center > 255 - threshold) ? 255 : center + threshold;
		bool isCorner = false;
		bool CornerType = false;
		for (int i = 0; i < 27; ++i)
		{
			int x = offsetX[i] + cX, y = offsetY[i] + cY;
			highCount = (tile[x][y] > t_high) ? highCount + 1 : 0;
			lowCount = (tile[x][y] < t_low) ? lowCount + 1 : 0;
			if (highCount >= 12)
			{

				isCorner = true;
				CornerType = true;
			}
			else if (lowCount >=12)
			{
				isCorner = true;
				CornerType = false;
			}
		}
		cornerMap[source] = isCorner ? (CornerType ? 255 : 128) : 0;
	}
}


#define tile 5

__device__ void localElMul(float in1[tile][tile], float in2[tile][tile], float out[tile][tile])
{
	for (int i = 0; i<tile; ++i)
		for (int j = 0; j < tile; ++j)
		{
			out[i][j] = in1[i][j] * in2[i][j];
		}
}


__global__ void FAST_Refine(unsigned char* __restrict__ inputImage, uint4* __restrict__ cornerMap,const int count, const int width, const  int height)
{
	const float k = 0.06;
	const float gk3[][3] = { {1,2,1},{2,4,2},{1,2,1} };
	float roi[tile][tile];
	float ix[tile][tile]; 
	float  iy[tile][tile];
	float  ixx[tile][tile];
	float  iyy[tile][tile]; 
	float  ixy[tile][tile];
	int threadIndex = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadIndex < count)
	{
		uint4 cvector = cornerMap[threadIndex];
		if (cvector.x > 3 && cvector.x < width - 3 && cvector.y>3 && cvector.y < height - 3)
		{
			int ptr = cvector.x - (tile / 2) + (cvector.y - tile / 2)*width;
			for (int i = 0; i < tile; ++i)
			{
				for (int j = 0; j < tile; ++j)
				{
					roi[i][j] = inputImage[ptr];
					inputImage[ptr] = 0;
					ptr += 1;
				}
				ptr += width - tile;
			}

			for (int i = 0; i < tile; ++i)
			{
				for (int j = 1; j < (tile - 2); ++j)
				{
					ix[i][j] = roi[i][j + 1] - roi[i][j - 1];
					iy[j][i] = roi[j + 1][i] - roi[j - 1][i];
				}
			}
			localElMul(ix, iy, ixy);
			localElMul(ix, ix, ixx);
			localElMul(iy, iy, iyy);

			float sxx = 0, syy = 0, sxy = 0;
			for (int i = 1; i <= 3; ++i)
			{
				for (int j = 1; j <= 3; ++j)
				{
					sxx += ixx[i][j] * gk3[i][j];
					sxy += ixy[i][j] * gk3[i][j];
					syy += iyy[i][j] * gk3[i][j];
				}
			}

			sxx /= 16;
			syy /= 16;
			sxy /= 16;
			float trace = sxx + sxy;
			float trace2 = trace*trace;
			float det = sxx*syy - sxy*sxy;
			cornerMap[threadIndex].w = (det - trace2* k);
		}
		else
		{
			cornerMap[threadIndex].w = 0;
		}
	}	
	
	

	
}



