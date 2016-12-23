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
		cudaMemset(ptr, value, sizeof(T));
	}
	~cuArray()
	{
		cudaFree(ptr);
	}
	void clear()
	{
		cudaMemset(ptr, 0, size*sizeof(T));
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
		cudaMemcpy(ptr, source, size * sizeof(T), cudaMemcpyHostToDevice);
	}
	void download(T* dest)
	{
		cudaMemcpy(dest, ptr, size*sizeof(T), cudaMemcpyDeviceToHost);
	}
	void upload(T* source, size_t length)
	{
		if(size>length)
			cudaMemcpy(ptr, source, length * sizeof(T), cudaMemcpyHostToDevice);
	}
	void download(T* dest, size_t length)
	{
		if (size>length)
			cudaMemcpy(dest, ptr, length* sizeof(T), cudaMemcpyDeviceToHost);
	}

protected:
	T* ptr;
	int size;
};

#endif