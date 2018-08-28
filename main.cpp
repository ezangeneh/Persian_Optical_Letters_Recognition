#include <QCoreApplication>
#include "farsi_ocr/letterRecognizer.h"
#include "tiny-cnn/tiny_cnn.h"
#include "opencv2/xfeatures2d.hpp"

using namespace tiny_cnn;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QString TrainPath = "db_letter_train";
    QString TrainPath2 = "Train2_DB";//for aggregation
    QString TestPath = "Letters_DB";
    QDir dir(TestPath);
    QStringList imageNames = dir.entryList(QDir::Filter::Files);

    letterRecognizer dr;

// MLP & CNN & HOG

    //TRAIN
//    qDebug()<<"Start Of CNN Training";
//    dr.TRAINCNN(cv::Size(32,32),TrainPath);
//    qDebug()<<"Start Of MLP Training";
//    dr.TRAINMLP(cv::Size(32,32),TrainPath);
    qDebug()<<"Start Of HOG Feature Extraction";
    dr.TrainandExtractHogFeature(cv::Size(32,32),TrainPath);


//    qDebug()<<"Start Of Feature Aggregation";
//    dr.aggregate(cv::Size(32,32),TrainPath2);


    qDebug()<<"Start of Testing";

    dr.loadCNNmodel("CNN_letters");
    dr.loadMLPmodel("MLP_letters");
    dr.loadMLPAGGmodel("mlp_aggregate");


    int testSize = 100;

    //Test
    unsigned int MLPcount = 0, CNNcount = 0, HOGcount = 0, Aggregatecount = 0;
    for(int i = 0; i < testSize; i++)
    {
        qDebug()<<i+1<<" Of "<<imageNames.size();
        QString filename = TestPath + QString("/") + imageNames[i];

        QString label_str;
        QChar label_str1 = filename[filename.size()-5];
        QChar label_str2 = filename[filename.size()-6];
        if(label_str2 != '_')
        {
            label_str = QString(label_str2) + QString(label_str1);
        }
        else
        {
            label_str = QString(label_str1);
        }

        label_t label_true = label_str.toInt();


        auto MLPlabel = dr.testMLP(filename,cv::Size(32,32));
        if(MLPlabel == label_true)
            ++MLPcount;

        auto HOGlabel = dr.testWithKnnOnHOG(cv::Size(32,32),filename);
        if(HOGlabel == label_true)
            ++HOGcount;

        auto CNNlabel = dr.testCNN(filename,cv::Size(32,32));
        if(CNNlabel == label_true)
            ++CNNcount;

        auto AggLabel = dr.testAggregation(filename,cv::Size(32,32));
        if(AggLabel == label_true)
            ++Aggregatecount;

    }

    qDebug()<<"Accuracy Of CNN = "<<(double)CNNcount/testSize;
    qDebug()<<"Accuracy Of MLP = "<<(double)MLPcount/testSize;
    qDebug()<<"Accuracy Of HOG = "<<(double)HOGcount/testSize;
    qDebug()<<"Accuracy Of Aggregation = "<<(double)Aggregatecount/testSize;

    return a.exec();
}
