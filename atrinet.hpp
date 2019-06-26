/***** we have to define USE_OPENCV before include opencv.hpp, 
or else, we can not use the function AddMatVector *****/

#ifndef USE_OPENCV
#define USE_OPENCV
#endif

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "opencv2/opencv.hpp"

#include "caffe/layers/memory_data_layer.hpp"
//#include "boost/shared_ptr.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <opencv/highgui.h>
//#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace caffe;
using namespace boost;

void copy(const float* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<float>(i, j) = *(data_src + i*img_dst.rows + j);
}

void copy(const Mat& img_src, float* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<float>(i, j);
}

void copy(const uchar* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<uchar>(i, j) = *(data_src + i*img_dst.rows + j);
}
void copy(const Mat& img_src, uchar* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<uchar>(i, j);
}

double CosineSimilarity(Mat mVec1, Mat  mVec2)
{
	return (mVec1.dot(mVec2)) / (sqrt(mVec1.dot(mVec1))*sqrt(mVec2.dot(mVec2)));
}
