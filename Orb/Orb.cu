#include "Orb.h"
#include "FAST.cuh"
#include "Profiler.h"
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

std::vector<float4> Orb::fast(cuArray<uchar>& ibuffer, cuArray<uchar>& aux, int thres, const int arc_length, const int width, const int height, const bool supression, const int padding)
{
	std::vector<float4> corners;
	cv::Mat auxmat = cv::Mat(width, height, CV_8UC1);
	FAST << < dim3(width / FAST_TILE, height / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >> > (ibuffer, aux, thres, arc_length, width, height);
	aux.download(auxmat.data);
	int x = padding + padding*width;
	for (uint i = padding; i < height - padding; ++i,x += padding * 2)
	{
		for (uint j = padding; j < width - padding; ++j,++x)
		{
			uint cvalue = auxmat.data[x];
			if (cvalue > 0)
			{
				corners.push_back({ (float)j,(float)i,0,0 });
			}
		}
	}
	if (supression)
	{
		AngleMap.upload(corners.data(), corners.size());
		FAST_Refine << < corners.size() / 32, 32 >> >(ibuffer, AngleMap, corners.size(), width, height);
		AngleMap.download(corners.data(), corners.size());
		std::sort(corners.begin(), corners.end(), [](float4& c1, float4& c2) {
			return c1.w > c2.w;
		});
		int minc = corners.size() >= 4000 ? 4000 : corners.size();
		std::vector<float4> strong = std::vector<float4>(corners.begin(), corners.begin() + minc);
		return strong;
	}
	else
	{
		return corners;
	}


}
