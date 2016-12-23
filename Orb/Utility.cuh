#pragma once
#ifndef UTILITY_CUH
#define UTILITY_CUH

#include "cuda_runtime.h"

__global__ void OrientFast(unsigned char* InputImage, uint4*  CornerMap, const int width, const int height, const int length);

__global__ void rBRIEF(unsigned char* InputImage, uint4* __restrict__ CornerMap, uint4* Feature_Map, const int width, const int height);




#endif