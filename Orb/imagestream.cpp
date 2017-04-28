#include "imagestream.h"
#include <Windows.h>
#include <string>
#include <opencv2\highgui.hpp>
#include <iostream>
imagestream::imagestream(char* directory) : imagestream(directory, "*")
{

}
imagestream::imagestream(char* directory, std::string extension)
{
	_directory = directory;
	_extension = extension;
	_files = getFiles(query_compute(std::string(directory), extension));
}

imagestream::imagestream(std::string directory) : imagestream(directory, "*")
{

}

imagestream::imagestream(std::string directory, std::string extension)
{
	_directory = directory;
	_extension = extension;
	_files = getFiles(query_compute(directory, extension));
}
std::string imagestream::query_compute(std::string directory, std::string extension)
{
	return directory + "*." + extension;
}
void imagestream::reload()
{
	if (_files.size() > 0)
	{
		idx = 0;
	}
}

int imagestream::size()
{
	return _files.size();
}

std::vector<std::string> imagestream::getFiles(std::string directory) {
	std::vector<std::string> files;
	WIN32_FIND_DATA fileData;
	HANDLE hFind;

	if (!((hFind = FindFirstFile(directory.c_str(), &fileData)) == INVALID_HANDLE_VALUE))
	{
		while (FindNextFile(hFind, &fileData))
		{
			files.push_back(fileData.cFileName);
		}
	}

	FindClose(hFind);
	return files;
}
bool imagestream::read(cv::Mat& mat)
{

	if (idx<_files.size() && _files.size()>0)
	{
		mat = cv::imread(_directory + _files[idx]);
		++idx;
		return true;
	}
	return false;
}
bool imagestream::operator >> (cv::Mat& mat)
{

	if (idx<_files.size() && _files.size()>0)
	{
		mat = cv::imread(_directory + _files[idx]);
		++idx;
		return true;
	}
	return false;
}
bool imagestream::operator >> (std::string& s)
{

	if (idx<_files.size() && _files.size()>0)
	{
		s = _directory + _files[idx];
		++idx;
		return true;
	}
	return false;
}
void imagestream::setDirectory(std::string directory)
{
	_files = getFiles(query_compute(directory, _extension));
	reload();
}

void imagestream::setDirectory(char* directory)
{
	_files = getFiles(query_compute(std::string(directory), _extension));
	reload();
}