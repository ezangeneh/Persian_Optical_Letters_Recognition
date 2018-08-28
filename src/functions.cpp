#include "functions.h"
//
//#include "boost/filesystem/operations.hpp"
//#include "boost/filesystem/path.hpp"
//#include "boost/progress.hpp"
//namespace fs = boost::filesystem;

using namespace cv;

// -----------------------------------------------------------------------------------------------------------------------------------
// Function: Returns the precision of an input number of type double with num numbers of precision.
//      Input 1  (double inp):			     Input rectangle 1.
//      Input 2  (int num):		             Input rectangle 2.
double precisionOf(double inp, int num)
{
    double mul = pow(10,num);
    int holder = inp * mul;
    return (double)holder/mul;
}

//
//void readDirectory(string dir,vector<string> &names ,bool parent)
//{
//    fs::path full_path(dir);
//    if ( fs::is_directory( full_path ) )
//     {
//       fs::directory_iterator end_iter;
//
//       for ( fs::directory_iterator dir_itr( full_path );
//             dir_itr != end_iter;
//             ++dir_itr )
//       {
//
//
//      //     if ( fs::is_regular_file( dir_itr->status() ) )
//      //     {
//               names.push_back(dir_itr -> path().string());
//               sort(names.begin(),names.end());
//       //    }
//
//
//
//       }
//    }
//}


string num2str(double number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}



// bernoson threshold
//



void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    int len = std::max(src.cols, src.rows);
    Point2f pt(len / 2., len / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

    cv::warpAffine(src, dst, r, cv::Size(len, len));
}

// ... ]

//
