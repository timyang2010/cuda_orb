#pragma once
#include <chrono>
#include <string>
#include <queue>

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
	void Log(std::string name)
	{
		long long t = Stop();
		logs.push(record(name, t));
		Start();
	}
	void Report()
	{
		while (!logs.empty())
		{
			record rd = logs.front();
			std::cout << rd.name << ": " << rd.duration << "us  ";
			logs.pop();
		}
		std::cout << '\r';
	}
protected:
	int count;
	std::chrono::time_point<std::chrono::steady_clock> time;
	struct record
	{
		record(std::string _name, long long _duration) 
		{
			name = _name;
			duration = _duration;
		}
		std::string name;
		long long duration;
	};
	std::queue<record> logs;
	
};
