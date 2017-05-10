#pragma once

#include "cuda_runtime.h"
#include "BRIEF.h"
#define FAST_TILE 16
__global__ void FAST(unsigned char* , unsigned char*, const int, const int, const int, const int);
__global__ void FAST_Refine(unsigned char*, ty::Keypoint*, const int, const int, const int);
__global__ void ComputeOrientation(unsigned char*, ty::Keypoint*, const int, const int, const int);
