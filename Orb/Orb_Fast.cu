#include "Orb.h"
#include "FAST.cuh"

#include <arrayfire.h> 

std::vector<float4> Orb::fast(cuArray<uchar>& ibuffer, cuArray<uchar>& aux, const int width, const int height,  const int padding)
{
	std::vector<float4> corners;
	cv::Mat auxmat = cv::Mat(width, height, CV_8UC1);
	FAST << < dim3(width / FAST_TILE, height / FAST_TILE), dim3(FAST_TILE, FAST_TILE) >> > (ibuffer, aux, 50, width, height);
	aux.download(auxmat.data);
	for (uint i = padding; i < width - padding; ++i)
		for (uint j = padding; j < height - padding; ++j)
		{
			uint cvalue = auxmat.data[i + j*width];
			if (cvalue > 0)
			{
				corners.push_back({ (float)i,(float)j,0,0 });
			}
		}
	return corners;
}
void AFFAST(cv::Mat& grey, std::vector<float4>& poi)
{
	af::array afa = transpose(af::array(grey.cols, grey.rows, grey.data, af::source::afHost));
	af::features fast_features = af::fast(afa, 30, 12, true, 0.01f);
	int N = fast_features.getNumFeatures();
	af::array x_pos = fast_features.getX();
	af::array y_pos = fast_features.getY();
	af::array scores = fast_features.getScore();
	float* x = x_pos.host<float>();
	float* y = y_pos.host<float>();
	//float* x = x_pos.host<float>();
	for (int i = 0; i < N; ++i)
	{
		poi.push_back(float4{ (float)x[i],(float)y[i], 0,0 });
	}

}

