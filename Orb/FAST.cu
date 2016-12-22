
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

		for (int i = 0; i < 27; ++i)
		{
			int x = offsetX[i] + cX, y = offsetY[i] + cY;
			highCount = (tile[x][y] > t_high) ? highCount + 1 : 0;
			lowCount = (tile[x][y] < t_low) ? lowCount + 1 : 0;
			if ((highCount >= 9 && highCount < 13) || (lowCount >= 9 && lowCount < 13))
			{
				isCorner = true;
			}
			else if (highCount >= 13 || lowCount >= 13)	//reject noise pixel & sharp corner (must not replace this with an "else" or corner test on some orientation might fail)
			{
				isCorner = false;
			}
		}
		cornerMap[source] = isCorner ? 255 : 0;
	}
}

