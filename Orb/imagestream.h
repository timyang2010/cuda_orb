#pragma once
#ifndef IMAGESTREAM_H
#define IMAGESTREAM_H

#include <opencv2\core.hpp>
#include <vector>
class imagestream
{
public:
	imagestream(char* dir);
	imagestream(char* dir, std::string extension);
	imagestream(std::string dir);
	imagestream(std::string dir, std::string extension);
	void setDirectory(char* dir);
	void setDirectory(std::string dir);
	bool operator >> (cv::Mat& mat);
	bool read(cv::Mat& mat);
	bool operator >> (std::string& str);
	int size();
protected:
	std::string query_compute(std::string dir, std::string ext);
	std::vector<std::string> getFiles(std::string directory);
private:
	void reload();
	std::vector<std::string> _files;
	std::string _directory;
	std::string _extension;
	int idx;
};


#endif