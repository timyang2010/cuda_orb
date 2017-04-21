#pragma once
#include <chrono>
#include <string>
#include <queue>
#include <sstream>

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
	void Message(std::string msg, float value)
	{
		logs.push(record(msg, value,true));
	}
	void Report()
	{
		
		int sum = 0;
		while (!logs.empty())
		{
			record rd = logs.front();
			if (rd.is_msg)
			{
				std::cout << rd.name << ": " << rd.value << "    ";
			}
			else 
			{
				sum += rd.duration;
				std::cout << rd.name << ": " << rd.duration << "us  ";
			
			}
			logs.pop();
		}
		std::cout << "sum:" << sum << "us  ";
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
			is_msg = false;
		}
		record(std::string _name, float _value,bool _is_msg)
		{
			name = _name;
			value = _value;
			duration = 0;
			is_msg = _is_msg;
		}
		std::string name;
		long long duration;
		float value;
		bool is_msg;
	};
	std::queue<record> logs;
	
};
