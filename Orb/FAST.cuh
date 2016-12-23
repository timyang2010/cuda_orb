#pragma once

#include "cuda_runtime.h"
#define FAST_TILE 16
__global__ void FAST(unsigned char* , unsigned char*, const int, const int, const  int);

