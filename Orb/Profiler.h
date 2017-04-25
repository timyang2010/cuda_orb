#pragma once

#include <string>
#include <queue>
#include <sstream>
#include <chrono>
class Profiler
{
public:
	Profiler();
	void Start();
	int Count();
	long long Stop();
	void Log(std::string name);
	void Message(std::string msg, float value);
	void Report();

protected:
	int count;
	std::chrono::time_point<std::chrono::steady_clock> time;
	struct record
	{
		record(std::string _name, long long _duration);
		record(std::string _name, float _value, bool _is_msg);
		std::string name;
		long long duration;
		float value;
		bool is_msg;
	};
	std::queue<record> logs;	
};
