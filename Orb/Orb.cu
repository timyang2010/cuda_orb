#include "Orb.h"
#include "FAST.cuh"

#include <arrayfire.h> 
using namespace af;
Orb::Orb()
{

}
void Orb::computeOrientation(cuArray<unsigned char>& frame, std::vector<float4>& corners, int width, int height)
{
	int cc = corners.size() < CORNER_LIMIT ? corners.size() : CORNER_LIMIT;
	AngleMap.upload(corners.data(), cc);
	ComputeOrientation << < corners.size() / 32, 32 >> > (frame, AngleMap, cc, width, height);
	AngleMap.download(corners.data(), cc);
}
