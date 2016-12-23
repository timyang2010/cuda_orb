// bivariate_gauss.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <random>
#include <string>
#include <sstream>
#include <iostream>
using namespace std;


int main()
{
	stringstream ssx, ssy;
	for (int i = 0; i < 512; ++i)
	{
		ssx << rand() % 31 << ", ";
		ssy << rand() % 31 << ", ";
	}



	cout << ssx.str() << endl;
	cout << ssy.str() << endl;

	int g;
	cin >> g;
    return 0;
}

