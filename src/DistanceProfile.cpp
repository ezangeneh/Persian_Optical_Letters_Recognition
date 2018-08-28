#include "DistanceProfile.h"
#include "functions.h"
#include <opencv2/opencv.hpp>


using namespace cv;

DistanceProfile::DistanceProfile()
{
}


void DistanceProfile::compute(Mat &in, Mat &out)
{
    resize(in, in, Size(64, 64));
//	in = thresh_bernsen(in, 64, 50);
	threshold(in, in, 128, 255, THRESH_BINARY_INV);

	Mat o(4, in.size().height, CV_32F), features(64, 4, CV_32F),ff(1,64*4,CV_32F);
	double min,max;
	Point minIdx,maxIdx;
	for (int j = 0; j < 4; j++)
	{
		rotate(in, 90, in);
		for (size_t i = 0; i < 64; i++)
		{
			minMaxLoc(in.row(i), &min, &max, &minIdx, &maxIdx);
			features.at<float>(i, j) = (float)maxIdx.x;
		}
	}
	out = features;
}

DistanceProfile::~DistanceProfile()
{
}
