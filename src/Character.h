#ifndef CHARACTER_H
#define CHARACTER_H


#include "opencv/cv.h"

//using namespace cv;

class Character
{
public:
    cv::Rect bBox;  // bounfing box of letter in license plate
	float rScore; // score of classifier fo this letter
	float dScore; //score of detection
    cv::Mat img; // Patch of letter in license plate
	bool isNumber;  // the digit is number or a letter
	int idx; // indx position of digit in license plate
//	union
//	{
		int digit_detected;
		char char_detected;
//	}
	
    bool operator < (const Character &) const;
    Character();
    ~Character();
};

#endif
