#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <opencv2/ml/ml.hpp>  
#include <opencv2/video/video.hpp>  
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

#ifdef WIN32
#pragma comment(lib, "OpenCV.lib")
#ifdef _DEBUG

#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#pragma comment(lib, "opencv_objdetect2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "opencv_contrib2413d.lib")
#pragma comment(lib, "opencv_video2413d.lib")
#pragma comment(lib, "opencv_gpu2413d.lib")
#pragma comment(lib, "opencv_ocl2413d.lib")
#pragma comment(lib, "opencv_ml2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_nonfree2413d.lib")
#pragma comment(lib, "opencv_flann2413d.lib")
#pragma comment(lib, "opencv_calib3d2413d.lib")
#endif

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "vfw32.lib") 

#ifdef NDEBUG
#pragma comment(lib, "opencv_core2413.lib")
#pragma comment(lib, "opencv_highgui2413.lib")
#pragma comment(lib, "opencv_objdetect2413.lib")
#pragma comment(lib, "opencv_imgproc2413.lib")
#pragma comment(lib, "opencv_contrib2413.lib")
#pragma comment(lib, "opencv_video2413.lib")
#pragma comment(lib, "opencv_gpu2413.lib")
#pragma comment(lib, "opencv_ocl2413.lib")
#pragma comment(lib, "opencv_ml2413.lib")
#pragma comment(lib, "opencv_features2d2413.lib")
#pragma comment(lib, "opencv_nonfree2413.lib")
#pragma comment(lib, "opencv_flann2413.lib")
#pragma comment(lib, "opencv_calib3d2413.lib")
#endif
#endif

void Copy(const float* data_src, Mat& img_dst);
void Copy(const Mat& img_src, float* data_dst);
void Copy(const uchar * data_src, Mat& img_dst);
void Copy(const Mat& img_src, uchar* data_dst);
void Copy(const Mat& src, Mat& dst, const Mat& mask, int offsetx, int offsety);

double CosineSimilarity(Mat mVec1, Mat  mVec2);
Mat Rotate(const Mat& src, double angle, Scalar border = Scalar(0), Point2f center = Point2f(0.0, 0.0));
void PrintText(Mat& img, string text, cv::Point center, int font, double scale, Scalar color, int thickness);
void DrawImage2(Mat& img, void* hdc, cv::Rect rect);
vector<cv::Point> FindPoint(const Mat& img);