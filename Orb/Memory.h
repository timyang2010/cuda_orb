#pragma once
#ifndef MEMORY_H
#define MEMORY_H
#include "cuda_runtime.h"
template<typename T> class cuArray
{
public:
	cuArray()
	{
		size = 0;
	}
	cuArray(int _size)
	{
		cudaMalloc(&device_ptr, _size * sizeof(T));
		size = _size;
	}
	cuArray(int _size,int value)
	{
		cudaMalloc(&device_ptr, _size * sizeof(T));
		size = _size;
		cudaMemset(device_ptr, value, sizeof(T));
	}

	~cuArray()
	{
		if (size>0)
			cudaFree(device_ptr);
	}

	cuArray<T> cuArray<T>::operator=(cuArray<T>& rhs)
	{
		device_ptr = rhs.device_ptr;
		size = rhs.size;
		return *this;
	}

	void clear()
	{
		cudaMemset(device_ptr, 0, size*sizeof(T));
	}
	operator T* () const
	{
		return device_ptr;
	}
	operator void* () const
	{
		return device_ptr;
	}
	virtual void upload(T* source)
	{
		cudaMemcpy(device_ptr, source, size * sizeof(T), cudaMemcpyHostToDevice);
	}
	virtual void download(T* dest)
	{
		cudaMemcpy(dest, device_ptr, size*sizeof(T), cudaMemcpyDeviceToHost);
	}
	virtual void upload(T* source, size_t length)
	{
		if(size>length)
			cudaMemcpy(device_ptr, source, length * sizeof(T), cudaMemcpyHostToDevice);
	}
	virtual void download(T* dest, size_t length)
	{
		if (size>length)
			cudaMemcpy(dest, device_ptr, length* sizeof(T), cudaMemcpyDeviceToHost);
	}
	int length()
	{
		return size;
	}
protected:
	T* device_ptr;
	int size;
};
//convert pointer to double pointer with respect to width and height
template<typename T> T** convert2D(T* in, unsigned int width, unsigned int height)
{
	T** a = new T*[height];
	for (int i = 0, j = 0; i < height; ++i)
	{
		a[i] = &(in[j]);
		j += width;
	}
	return a;
}
#endif