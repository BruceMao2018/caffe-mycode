#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat src = imread("test2.png");
	if( !src.data ) { cout << "read img error" << endl; return -1; }
	imshow("原始图片", src);

	//最简单的canny用法，拿到原图后直接用
	Mat src2;
	Canny(src, src2, 150, 100, 3);
	imshow("canny边缘检测效果图", src2);

	//高阶的canny用法，转成灰度图，降噪，用canny，最后将得到的边缘作为掩码，拷贝原图到效果图上，得到彩色的边缘图
	Mat dst, edge, gray;
	Mat src1 = src.clone();

	//创建与src同类型和大小的矩阵(dst)
	dst.create(src1.size(), src1.type());

	//将原图像转换为灰度图像
	cvtColor(src1, gray, CV_BGR2GRAY);

	//先用使用 3x3内核来降噪
	blur(gray, edge, Size(3, 3));

	//运行Canny算子进行边缘检测
	Canny(edge, edge, 3, 9, 3);

	//将g_dstImage内的所有元素设置为0 
	dst = Scalar::all(0);

	//使用边缘图edge作为掩码,将原图拷贝到dst中
	src1.copyTo( dst, edge );

	imshow("边缘检测效果图2", dst);



	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst2;
	Mat src3 = src.clone();

	//求ｘ方向梯度
	Sobel( src3, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("x方向Sobel", abs_grad_x);

	//求Y方向梯度
	Sobel( src3, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("y方向Sobel", abs_grad_y);

	//合并梯度
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst2);
	imshow("整体方向Sobel", dst2);

	waitKey(0);
	return 0;
}


/*
	其他边缘检测函数
	Laplace
	Laplacian
	Scharr
*/
