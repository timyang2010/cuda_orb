#pragma once
#include "Orb.h"
#include <string>
void TrackCamera(std::string arg);
void BRIEF_Optimize(int argc,char** argv);
std::vector<Orb::Feature> TrackKeypoints(cv::Mat& frame, Orb& orb, Orb::MODE mode = Orb::MODE::MODE_RBRIEF,int max_keypoints = 1500);