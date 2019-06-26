#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat srcImage, dstImage;
	Mat map_x, map_y;

	srcImage = imread("test5.png");
	if ( !srcImage.data ) { cout << "read image error" << endl; return -1; }

	imshow("原始图像", srcImage);

	//创建与原始图片一样的效果图, ｘ重映射图, Y重映射图
	dstImage.create(srcImage.size(), srcImage.type());
	map_x.create(srcImage.size(), CV_32FC1);
	map_y.create(srcImage.size(), CV_32FC1);

	namedWindow("第一种重映射效果图", CV_WINDOW_AUTOSIZE);

	//双层循环，遍历每一个像素点，改变map_x & map_y的值

	//第一种映射
/*
	for (int i = 0; i < srcImage.rows; i++)
	{
		for( int j = 0; j < srcImage.cols; j++)
		{
			if( j > srcImage.cols*0.25 && j < srcImage.cols*0.75 && i > srcImage.rows * 0.25 && i < srcImage.rows * 0.75)
			{
				map_x.at<float>(i, j) = static_cast<float>(2*( j - srcImage.cols * 0.25 ) + 0.5);
				map_y.at<float>(i, j) = static_cast<float>(2*( i - srcImage.rows * 0.25 ) + 0.5);
			}
			else
			{
				map_x.at<float>(i, j) = 0;
				map_y.at<float>(i, j) = 0;
			}
		}
	}
*/
	for( int j = 0; j < srcImage.rows;j++)
	{ 
		for( int i = 0; i < srcImage.cols;i++)
		{
				if( i > srcImage.cols*0.25 && i < srcImage.cols*0.75 && j > srcImage.rows*0.25 && j < srcImage.rows*0.75)
				{
					map_x.at<float>(j,i) = static_cast<float>(2*( i - srcImage.cols*0.25 ) + 0.5);
					map_y.at<float>(j,i) = static_cast<float>(2*( j - srcImage.rows*0.25 ) + 0.5);
				}
				else
				{ 
					map_x.at<float>(j,i) = 0;
					map_y.at<float>(j,i) = 0;
				}
		}
	}
	remap( srcImage, dstImage, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
	imshow("第一种重映射效果图", dstImage);

	waitKey(0);
	return 0;
}
