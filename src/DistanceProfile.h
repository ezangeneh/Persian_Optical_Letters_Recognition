#pragma once
#include "opencv/highgui.h"

//using namespace cv;
class DistanceProfile
{
public:
    void compute(cv::Mat &, cv::Mat &);
	DistanceProfile();
	~DistanceProfile();
};

