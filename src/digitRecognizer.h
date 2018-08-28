#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <QDir>
#include <QStringList>
#include <QDebug>
#include "../tiny-cnn/tiny_cnn.h"

//using namespace cv;
using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

//enum train_type{ TRAIN_HOG, TRAIN_DCT, TRAIN_RAW, TRAIN_DISTANCE_PROFILE, TRAIN_MLP, TRAIN_CNN };

class digitRecognizer
{
    vector<vector <cv::Mat> > digitTemps;  // template of digits
    cv::Ptr<cv::ml::KNearest> KNNClassify;
    cv::Size tempSize;
    cv::LDA lda;

    std::vector<cv::Mat> trainData;
    std::vector<int> trainLabel;

public:
    void TRAIN(cv::Size tempSize, QString TrainPath);
    void FeatureExtractionCNN(cv::Size tempSize, QString TrainPath);
    int testWithKnn(cv::Size tempSize, QString filename);
    size_t test(QString filename, cv::Size TempSize);
    void tightImage(cv::Mat &in,cv::Mat &out);
    string getAlphabetByID(int);
    cv::Mat image2mat(image<unsigned char>& img);
	digitRecognizer();
	~digitRecognizer();

};
