#pragma once
#include "Orb.h"
#include <string>
void TrackCamera(std::string arg);
void BRIEF_Optimize(int argc,char** argv);
std::vector<Orb::Feature> TrackKeypoints(cv::Mat& frame, Orb& orb,int max_keypoints = 1500);