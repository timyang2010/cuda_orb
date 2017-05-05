
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include "Profiler.h"
#include "Memory.h"
#include "Orb.h"

#include <sstream>
#include <fstream>
#include "Application.h"
using namespace cv;
using namespace std;



int main(int argc,char** argv)
{
	string path;
	switch (argv[1][0])
	{
	case 't':
		BRIEF_Optimize(argv[2]);
		break;
	case 'c':
		fstream f(argv[2]);
		for (; getline(f, path);)
		{
			TrackCamera(path.c_str(), Orb::fromFile("pat.txt"));
		}
		break;

	}
	return 0;
}