#pragma once
#include <chrono>
class Profiler
{
public:
	Profiler()
	{
		count = -1;
	}
	void Start()
	{
		time = std::chrono::high_resolution_clock::now();
	}
	int Count()
	{
		++count;
		return count;
	}
	long long Stop()
	{

		auto elapsed = std::chrono::high_resolution_clock::now() - time;
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		return microseconds;
	}
protected:
	int count;
	std::chrono::time_point<std::chrono::steady_clock> time;
};
