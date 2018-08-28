#include "digitRecognizer.h"
#include "DistanceProfile.h"
#include "functions.h"
#include "cvBlobsLib/BlobResult.h"
#include "cvBlobsLib/blob.h"
#include "dirent.h"
#include "../tiny-cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace cv;


int eppochNum = 0;

//CvANN_MLP nnetworkDigit;
//CvANN_MLP nnetworkAll;

string convertLet2Dig(label_t lb);
digitRecognizer::digitRecognizer()
{

    KNNClassify->create();
}

void digitRecognizer::TRAIN( cv::Size tempSize, QString TrainPath)
{
    // sakhte shabake CNN

    network<mse, adagrad> cnn_digits;
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    // construct nets
    cnn_digits << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out

    //
    cnn_digits.init_weight();

    this->tempSize = tempSize;

    int width = tempSize.width;

    //khandane kol tasavir
    QDir dir(TrainPath);
    QStringList imageNames = dir.entryList(QDir::Filter::Files);

    std::vector<label_t> labels(imageNames.size());
    std::vector<vec_t> samples(imageNames.size());

    QList<int> values;
    for(int i=0;i<imageNames.size();i++)
        values.push_back(i);

    for(int i = 0; i < imageNames.size(); i++)//baraye har tasvire train
    {
        qDebug()<<"Read Data "<<i<<" From "<<imageNames.size();
        QString filename = TrainPath + QString("/") + imageNames[i];
        QString label_str;

        //tashkhise label vaghee
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

        label_t label = label_str.toInt();

        //khandane khode tasvir
        Mat im = imread(filename.toStdString(),0);
//        cv::resize(im,im,cv::Size(width,width));

        if(im.cols > im.rows)
        {
            if(im.cols > width)
            {
                int rows = ((double)width/im.cols)*im.rows;
                cv::resize(im,im,cv::Size(width,rows));
            }
        }
        else
        {
            if(im.rows > width)
            {
                int cols = ((double)width/im.rows)*im.cols;
                cv::resize(im,im,cv::Size(cols,width));
            }
        }

        int x_sub = width - im.cols;
        int y_sub = width - im.rows;

        cv::Mat padded;
        padded.create(width, width, im.type());
        padded.setTo(cv::Scalar::all(0));
        im.copyTo(padded(cv::Rect(x_sub/2, y_sub/2, im.cols, im.rows)));
        im = padded;

        cv::resize(im,im,tempSize);
        vec_t sample;

        //convert image to a vector
        for (int i = 0; i < im.rows; i++)
        {
            for (int j = 0; j < im.cols; j++)
            {
                float t = (im.at<uchar>(i, j)/255.0)*(2)+(-1);//normalize kardane har pixel
                sample.push_back(t);
            }
        }

        // darham kardane dadehaye train
        int ind = rand()%(values.size());
        int index = values[ind];
        values.removeAt(ind);
        samples[index] = sample;
        labels[index] = label;
    }



    progress_display disp(samples.size());
    timer t;
    int minibatch_size = 20;
    int num_epochs = 5;

    cnn_digits.optimizer().alpha *= std::sqrt(minibatch_size);

       // create callback
       auto on_enumerate_epoch = [&](){
           std::cout<<"\neppoch Number : "<<++eppochNum<<std::endl;
       };

       auto on_enumerate_minibatch = [&](){
//           std::cout<<"In Batch"<<std::endl;
           disp += minibatch_size;
       };

    qDebug()<<"Training";
    cnn_digits.train(samples,labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);
    qDebug()<<"FinishTraining";

    //zakhireye vazn haye amozesh dade shode
    ofstream outf("CNN_digits");
    outf<<cnn_digits;
    outf.close();
}


bool isBigger(vector<float> a, vector <float > b)
{
	return max_element(a.begin(), a.end())> max_element(b.begin(), b.end());

}

cv::Mat digitRecognizer::image2mat(image<unsigned char> &img)
{
    cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
    cv::Mat resized;
    cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
    return resized;
}

void digitRecognizer::FeatureExtractionCNN(cv::Size tempSize, QString TrainPath)
{

    network<mse, adagrad> cnn_digits;
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    // construct nets
    cnn_digits << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out


    ifstream inf("CNN_digits",ios_base::in);
    inf>>cnn_digits;
    inf.close();


    QDir dir(TrainPath);
    QStringList imageNames = dir.entryList(QDir::Filter::Files);

    int la;

    for(int i = 0; i < imageNames.size(); i++)//baraye har tasvire train
    {
        qDebug()<<"Read Data "<<i<<" From "<<imageNames.size();

        QString filename = TrainPath + QString("/") + imageNames[i];

        QChar label_str = filename[filename.size()-5];
        la = (uchar)QString(label_str).toInt();


        cv::Mat im = cv::imread(filename.toStdString(),0);

        int width = tempSize.width;

        if(im.cols > im.rows)
        {
            if(im.cols > width)
            {
                int rows = ((double)width/im.cols)*im.rows;
                cv::resize(im,im,cv::Size(width,rows));
            }
        }
        else
        {
            if(im.rows > width)
            {
                int cols = ((double)width/im.rows)*im.cols;
                cv::resize(im,im,cv::Size(cols,width));
            }
        }

        int x_sub = width - im.cols;
        int y_sub = width - im.rows;

        cv::Mat padded;
        padded.create(width, width, im.type());
        padded.setTo(cv::Scalar::all(0));
        im.copyTo(padded(cv::Rect(x_sub/2, y_sub/2, im.cols, im.rows)));
        im = padded;

        int a;
//        vec_t sample;
        vec_t sample;
        sample.reserve(1024);
        //convert image to a vector
        for (int i = 0; i < im.rows; i++)
        {
            for (int j = 0; j < im.cols; j++)
            {
                float t = (im.at<uchar>(i, j)/255.0)*(2)+(-1);
                sample.push_back(t);
            }
        }
    //        vec_t sample = vec;

        auto res = cnn_digits.predict(sample);

        auto outOfLayer5 = cnn_digits[5]->output_to_image();

        //vec_t to vectorized mat
        cv::Mat m;
        for(int i=0; i<124; i++)
        {
            m.push_back(outOfLayer5.data()[i]);
        }


        trainData.push_back(m);
        trainLabel.push_back(la);

    }


}

int digitRecognizer::testWithKnn(Size tempSize, QString filename)
{
    // feed forward test image in cnn and extract feature
    network<mse, adagrad> cnn_digits;
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    // construct nets
    cnn_digits << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out


    ifstream inf("CNN_digits",ios_base::in);
    inf>>cnn_digits;
    inf.close();

    cv::Mat im = cv::imread(filename.toStdString(),0);

    int width = tempSize.width;

    if(im.cols > im.rows)
    {
        if(im.cols > width)
        {
            int rows = ((double)width/im.cols)*im.rows;
            cv::resize(im,im,cv::Size(width,rows));
        }
    }
    else
    {
        if(im.rows > width)
        {
            int cols = ((double)width/im.rows)*im.cols;
            cv::resize(im,im,cv::Size(cols,width));
        }
    }

    int x_sub = width - im.cols;
    int y_sub = width - im.rows;

    cv::Mat padded;
    padded.create(width, width, im.type());
    padded.setTo(cv::Scalar::all(0));
    im.copyTo(padded(cv::Rect(x_sub/2, y_sub/2, im.cols, im.rows)));
    im = padded;

    vec_t sample;

    //convert image to a vector
    for (int i = 0; i < im.rows; i++)
    {
        for (int j = 0; j < im.cols; j++)
        {
            float t = (im.at<uchar>(i, j)/255.0)*(2)+(-1);
            sample.push_back(t);
        }
    }

    auto res = cnn_digits.predict(sample);

    auto outOfLayer5 = cnn_digits[5]->output_to_image();

    //vec_t to vectorized mat
//    cv::Mat m(1,124,CV_8UC1);
//    for(int i=0; i<124; i++)
//    {
//        m.at<uchar>(i,1) = outOfLayer5.data()[i];
//    }

    cv::Mat m;
    for(int i=0; i<124; i++)
    {
        m.push_back(outOfLayer5.data()[i]);
    }

    // find nearest sample to test data

    int min = 124*255;
    int label = -1;

    for(int i=0; i<trainData.size(); i++)
    {
//        cv::Mat mm = trainData[i];
        int dis = cv::norm(trainData[i],m,NORM_L2);
        if(dis < min)
        {
            min = dis;
            label = trainLabel[i];
        }
    }

    return label;
}

size_t digitRecognizer::test(QString filename, cv::Size tempSize)
{
    network<mse, adagrad> cnn_digits;
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    // construct nets
    cnn_digits << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out


    ifstream inf("CNN_digits",ios_base::in);
    inf>>cnn_digits;
    inf.close();

    cv::Mat im = cv::imread(filename.toStdString(),0);

    int width = tempSize.width;

    if(im.cols > im.rows)
    {
        if(im.cols > width)
        {
            int rows = ((double)width/im.cols)*im.rows;
            cv::resize(im,im,cv::Size(width,rows));
        }
    }
    else
    {
        if(im.rows > width)
        {
            int cols = ((double)width/im.rows)*im.cols;
            cv::resize(im,im,cv::Size(cols,width));
        }
    }

    int x_sub = width - im.cols;
    int y_sub = width - im.rows;

    cv::Mat padded;
    padded.create(width, width, im.type());
    padded.setTo(cv::Scalar::all(0));
    im.copyTo(padded(cv::Rect(x_sub/2, y_sub/2, im.cols, im.rows)));
    im = padded;

    vec_t sample;

    //convert image to a vector
    for (int i = 0; i < im.rows; i++)
    {
        for (int j = 0; j < im.cols; j++)
        {
            float t = (im.at<uchar>(i, j)/255.0)*(2)+(-1);
            sample.push_back(t);
        }
    }
//        vec_t sample = vec;

    auto res = cnn_digits.predict(sample);
    label_t predicted_letters = max_index(res);
    return predicted_letters;
}

void digitRecognizer::tightImage(Mat &in, Mat &out)
{
	Mat im;
	in.copyTo(im);
	//im=255-im;
	//cvtColor(im,im,CV_);
	//imwrite("test.jpg",im);

	// im.copyTo(out);
	//im.convertTo(im,CV_32F);
	threshold(im, im, 128, 255, CV_THRESH_BINARY_INV);
	im.convertTo(im, CV_8UC1);
	//imwrite("test2.jpg",im);

	IplImage  img = (IplImage)im;

	//cvSaveImage("test3.jpg",&img);

	CBlob *b, *a;
	CBlobResult r;
	r = CBlobResult(&img);
	if (r.GetNumBlobs() != 0)
	{
		b = r.GetBlob(0);
		int numbers = r.GetNumBlobs();
		for (int i = 1; i<numbers; i++)
		{
			a = r.GetBlob(i);
			if (a->Area() >b->Area())
				b = a;
		}
		if (b->GetBoundingBox().width > 0 && b->GetBoundingBox().height > 0)
			in(b->GetBoundingBox()).copyTo(out);
		else
			in.copyTo(out);
	}
	else
	{
		in.copyTo(out);
	}
}

string digitRecognizer::getAlphabetByID(int i)
{
	switch (i)
	{
	case 1:
		return "a";
	case 2:
		return "b";
	case 3:
		return "t";		//return "X";
	case 4:
		return "j";
	case 5:
		return "d";
	case 6:
		return "si";	//return "C";
	case 7:
		return "sa";	//return "S";
	case 8:
		return "ta";	//return "T";
	case 9:
		return "k";
	case 10:
		return "gh";	//return "G";
	case 11:
		return "l";
	case 12:
		return "m";
	case 13:
		return "n";
	case 14:
		return "v";
	case 15:
		return "h";
	case 16:
		return "y";
	case 17:
		return "e";
	case 18:
		return "z";	// Defective
	default:
		return "";
	}
}

digitRecognizer::~digitRecognizer()
{
}

string convertLet2Dig(label_t lb){
	string str = "";
	if (lb == 11)
		str = "a";
	else if (lb == 12)
		str = "b";
	else if (lb == 13)
		str = "ta";
	else if (lb == 14)
		str = "j";
	else if (lb == 15)
		str = "d";
	else if (lb == 16)
		str = "si";
	else if (lb == 17)
		str = "sa";
	else if (lb == 18)
		str = "k";
	else if (lb == 19)
		str = "gh";
	else if (lb == 20)
		str = "l";
	else if (lb == 21)
		str = "m";
	else if (lb == 22)
		str = "n";
	else if (lb == 23)
		str = "v";
	else if (lb == 24)
		str = "h";
	else if (lb == 25)
		str = "y";
	else if (lb == 26)
		str = "e";
	else if (lb == 27)
		str = "z";
	return str;
}
