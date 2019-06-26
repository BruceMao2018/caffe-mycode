#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat srcImage = imread("test2.png");
	Mat dstImage;
	if( !srcImage.data ) { cout << "Read img error" << endl; return -1; }

	dstImage = Mat::zeros(srcImage.size(), srcImage.type());

	//设定对比度和亮度的初值
	int ContrastValue = 80;
	int BrightValue = -80;

	cout << "srcImage.rows: " << srcImage.rows << " srcImage.cols: " << srcImage.cols << " channels: " << srcImage.channels() << endl;

	for ( int i = 0; i < srcImage.rows; i++)
		for ( int j = 0; j < srcImage.cols; j++)
		{
			dstImage.at<Vec3b>(i,j)[0] = saturate_cast<uchar>((ContrastValue * 0.01) * ( srcImage.at<Vec3b>(i, j)[0]) + BrightValue );
			dstImage.at<Vec3b>(i,j)[1] = saturate_cast<uchar>((ContrastValue * 0.01) * ( srcImage.at<Vec3b>(i, j)[1]) + BrightValue );
			dstImage.at<Vec3b>(i,j)[2] = saturate_cast<uchar>((ContrastValue * 0.01) * ( srcImage.at<Vec3b>(i, j)[2]) + BrightValue );
		}


	imshow("Original", srcImage);
	imshow("New", dstImage);

	waitKey(0);

	return 0;
}

/*
saturate_cast是一个类模板，防止数据溢出
对比度与亮度的原理公式：
	g(x) = a * f(x) + b;

    参数f(x)表示源图像像素。
    参数g(x) 表示输出图像像素。
    参数a（需要满足a>0）被称为增益（gain），常常被用来控制图像的对比度。
    参数b通常被称为偏置（bias），常常被用来控制图像的亮度。
*/
