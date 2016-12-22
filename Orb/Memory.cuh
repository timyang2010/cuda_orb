#pragma once
#ifndef MEMORY_CUH
#define MEMORY_CUH
#include "cuda_runtime.h"
template<typename T> class cuArray
{
public:
	cuArray(int _size)
	{
		cudaMalloc(&ptr, _size * sizeof(T));
		size = _size * sizeof(T);
	}
	cuArray(int _size,int value)
	{
		cudaMalloc(&ptr, _size * sizeof(T));
		size = _size * sizeof(T);
		cudaMemset(ptr, value, size);
	}
	~cuArray()
	{
		cudaFree(ptr);
	}
	void clear()
	{
		cudaMemset(ptr, 0, size);
	}
	operator T* () const
	{
		return ptr;
	}
	operator void* () const
	{
		return ptr;
	}
	void upload(T* source)
	{
		cudaMemcpy(ptr, source, size, cudaMemcpyHostToDevice);
	}
	void download(T* dest)
	{
		cudaMemcpy(dest, ptr, size, cudaMemcpyDeviceToHost);
	}
protected:
	T* ptr;
	int size;
};

#endif