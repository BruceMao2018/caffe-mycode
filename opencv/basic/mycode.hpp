#ifndef MY_CODE_HPP__
#define MY_CODE_HPP__

#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void Copy(const float* data_src, Mat& img_dst);
void Copy(const Mat& img_src, float* data_dst);
void Copy(const uchar * data_src, Mat& img_dst);
void Copy(const Mat& img_src, uchar* data_dst);
void Copy(const Mat& src, Mat& dst, const Mat& mask, int offsetx, int offsety);
double CosineSimilarity(Mat mVec1, Mat  mVec2);

#endif
