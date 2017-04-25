#include "Profiler.h"
#include <iostream>
using namespace std;
using namespace std::chrono;
Profiler::Profiler()
{
	count = -1;
}
void Profiler::Start()
{
	time = high_resolution_clock::now();
}
int Profiler::Count()
{
	++count;
	return count;
}
long long Profiler::Stop()
{
	auto elapsed = high_resolution_clock::now() - time;
	long long microseconds = duration_cast<std::chrono::microseconds>(elapsed).count();
	return microseconds;
}
void Profiler::Log(std::string name)
{
	long long t = Stop();
	logs.push(record(name, t));
	Start();
}
void Profiler::Message(std::string msg, float value)
{
	logs.push(record(msg, value, true));
}
void Profiler::Report()
{

	int sum = 0;
	while (!logs.empty())
	{
		record rd = logs.front();
		if (rd.is_msg)
		{
			cout << rd.name << ": " << rd.value << "    ";
		}
		else
		{
			sum += rd.duration;
			cout << rd.name << ": " << rd.duration << "us  ";

		}
		logs.pop();
	}
	cout << "sum:" << sum << "us  ";
	cout << '\r';
}


Profiler::record::record(std::string _name, long long _duration)
{
	name = _name;
	duration = _duration;
	is_msg = false;
}
Profiler::record::record(std::string _name, float _value, bool _is_msg)
{
	name = _name;
	value = _value;
	duration = 0;
	is_msg = _is_msg;
}
