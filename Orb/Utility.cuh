#pragma once
#ifndef UTILITY_CUH
#define UTILITY_CUH

#include "cuda_runtime.h"

__global__ void OrientFast(unsigned char* InputImage, uint4*  CornerMap, const int width, const int height, const int length);



#endif