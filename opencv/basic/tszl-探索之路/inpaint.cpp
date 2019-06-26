#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat imgSrc = imread(argv[1]);
	if( !imgSrc.data ) { cout << "read img error" << endl; return -1; }
	imshow("原图", imgSrc);

	Mat imgGray;
	cvtColor(imgSrc, imgGray, CV_RGB2GRAY, 0);
	imshow("原图的灰度图", imgGray);

	//通过阀值处理生成mask
	int th = 240;
	Mat imgMask;
	//Mat imgMask = Mat::zeros(imgSrc.size(), CV_8UC1);//mask是与原图同样大小的单通道图像
	threshold(imgGray, imgMask, th, 255, CV_THRESH_BINARY);
	imshow("二值化后图片", imgMask);

	// //对Mask膨胀处理，增加Mask面积
	Mat Kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgMask, imgMask, Kernel);
	imshow("mask膨胀后图", imgMask);

	inpaint(imgSrc, imgMask, imgSrc, 5, INPAINT_TELEA);

	imshow("修复后的图", imgSrc);

	waitKey(0);
	return 0;
}
