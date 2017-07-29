## Orb Feature Detector
Partially CUDA implementation for ORB (Oriented FAST and Rotated BRIEF) feature detector.


#### Dependencies 

* C++ 11
* CUDA 8
* OpenCV 3.0+


#### Usage
* See example projects
* API usage
```cpp
#include "Orb.h"
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;
vector<Orb::Feature> TrackKeypoints(Mat& frame, Orb& orb, int max_keypoints = 2048)
{
	Mat grey;
	cvtColor(frame, grey, CV_BGR2GRAY);
	vector<ty::Keypoint> corners = orb.detectKeypoints(grey, 25, 12, max_keypoints);
	vector<Orb::Feature> features = orb.extractFeatures(grey, corners);
	return features;
}
int main()
{
    Orb orb;
    Mat frame_l = imread("path/to/image_l");
    Mat frame_r = imread("path/to/image_r");
    vector<Orb::Feature> features_l = TrackKeypoints(frame_l,orb);
    vector<Orb::Feature> features_r = TrackKeypoints(frame_r,orb);
    auto pairs = ty::BRIEF::matchFeatures_gpu(features_l, features_r, 64);

}

```

#### Paper
1. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In Computer Vision (ICCV), 2011 IEEE international conference on (pp. 2564-2571). IEEE.
