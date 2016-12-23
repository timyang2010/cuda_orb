#include "Utility.cuh"
#include "device_launch_parameters.h"
#include "math.h"
__global__ void OrientFast(unsigned char* InputImage, uint4*  CornerMap, const int width, const int height,const int length)
{
	const int radius = 8;
	int index = blockIdx.x*blockDim.x + threadIdx.x;


	if (index<length )
	{
		uint4 corner = CornerMap[index];
		int cX = corner.x, cY = corner.y;
		double gx = 0, gy = 0, mass = 0;

		int addr = (cY - radius)*width + cX - radius / 2;
		for (int i = -radius; i <= radius; ++i)
		{
			for (int j = -radius; j <= radius; ++j)
			{
				if ((i*i + j*j) < radius*radius)
				{
					int m = InputImage[addr];
					mass += m;
					gx += (cX + j)*m;
					gy += (cY + i)*m;
				}
				
				addr += 1;
			}
			addr += width - radius;
		}
		gx /= mass;
		gy /= mass;

		double d = sqrt(pow(gx - (double)cX, 2) + pow(gy - (double)cY, 2));

		if (d > 0)
		{
			gx = cX + (gx - (double)cX) * 20 / d;
			gy = cY + (gy - (double)cY) * 20 / d;
		}
		CornerMap[index].w = gx;
		CornerMap[index].z = gy;
		//CornerMap[index].w = (unsigned int)(atanf(gy / gx) * 180 / 3.14159);
	}
	

}

__global__ void BRIEF(unsigned char* InputImage, uint4* __restrict__ CornerMap, uint4* Feature_Map, const int width, const int height, const int length)
{
	const int radius = 5;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < length)
	{


	}


}