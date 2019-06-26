#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat srcImage, dstImage1, dstImage2, dstImage3, dstImage4;
	Mat map_x1, map_y1, map_x2, map_y2, map_x3, map_y3, map_x4, map_y4;

	srcImage = imread("test5.png");
	if ( !srcImage.data ) { cout << "read image error" << endl; return -1; }

	imshow("原始图像", srcImage);

	//创建与原始图片一样的效果图, ｘ重映射图, Y重映射图
	dstImage1.create(srcImage.size(), srcImage.type());
	dstImage2.create(srcImage.size(), srcImage.type());
	dstImage3.create(srcImage.size(), srcImage.type());
	dstImage4.create(srcImage.size(), srcImage.type());
	map_x1.create(srcImage.size(), CV_32FC1);
	map_y1.create(srcImage.size(), CV_32FC1);
	map_x2.create(srcImage.size(), CV_32FC1);
	map_y2.create(srcImage.size(), CV_32FC1);
	map_x3.create(srcImage.size(), CV_32FC1);
	map_y3.create(srcImage.size(), CV_32FC1);
	map_x4.create(srcImage.size(), CV_32FC1);
	map_y4.create(srcImage.size(), CV_32FC1);

	//双层循环，遍历每一个像素点，改变map_x & map_y的值

	//第一种映射
	for (int i = 0; i < srcImage.rows; i++)
		for( int j = 0; j < srcImage.cols; j++)
		{
			if( j > srcImage.cols*0.25 && j < srcImage.cols*0.75 && i > srcImage.rows * 0.25 && i < srcImage.rows * 0.75)
			{
				map_x1.at<float>(i, j) = static_cast<float>(2*( j - srcImage.cols * 0.25 ) + 0.5);
				map_y1.at<float>(i, j) = static_cast<float>(2*( i - srcImage.rows * 0.25 ) + 0.5);
			}
			else
			{
				map_x1.at<float>(i, j) = 0;
				map_y1.at<float>(i, j) = 0;
			}
		}

	//第二种映射
	for( int i = 0; i < srcImage.rows; i++)
		for( int j = 0; j < srcImage.cols; j++)
		{
			map_x2.at<float>(i, j) = static_cast<float>(j);
			map_y2.at<float>(i, j) = static_cast<float>(srcImage.rows - i);
		}

	//第三种映射
	for( int i = 0; i < srcImage.rows; i++)
		for( int j = 0; j < srcImage.cols; j++)
		{
			map_x3.at<float>(i, j) = static_cast<float>(srcImage.cols -j);
			map_y3.at<float>(i, j) = static_cast<float>(i);
		}

	//第四种映射
	for( int i = 0; i < srcImage.rows; i++)
		for( int j = 0; j < srcImage.cols; j++)
		{
			map_x4.at<float>(i, j) = static_cast<float>(srcImage.cols - j);
			map_y4.at<float>(i, j) = static_cast<float>(srcImage.rows - i);
		}
	

	remap( srcImage, dstImage1, map_x1, map_y1, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
	remap( srcImage, dstImage2, map_x2, map_y2, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
	remap( srcImage, dstImage3, map_x3, map_y3, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
	remap( srcImage, dstImage4, map_x4, map_y4, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

	imshow("第一种重映射效果图", dstImage1);
	imshow("第二种重映射效果图", dstImage2);
	imshow("第三种重映射效果图", dstImage3);
	imshow("第四种重映射效果图", dstImage4);

	waitKey(0);
	return 0;
}
