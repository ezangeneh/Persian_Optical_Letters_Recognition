#include <iostream>
#include <opencv2/opencv.hpp>

//using namespace cv;

using namespace std;

#ifndef FUNCTIONS
#define FUNCTIONS

double precisionOf(double inp, int num);
//void readDirectory(string,vector<string> &,bool);
string num2str(double number);


// bernoson threshold
//


cv::Mat thresh_bernsen(cv::Mat& gray, int ksize, int contrast_limit);

void rotate(cv::Mat& src, double angle, cv::Mat& dst);


// ... ]

//
#endif // FUNCTIONS

