#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat srcImg = imread("test1.bmp");
	if( !srcImg.data )
	{
		cout << "read img error" << endl;
		return -1;
	}

	//namedWindow("srcImg", WINDOW_NORMAL);
	imshow("原始图片", srcImg);

	Mat logo = imread("logo.bmp", IMREAD_COLOR);
	if( !logo.data )
	{
		cout << "read logo error" << endl;
		return -1;
	}

	Mat roi1 = srcImg(Rect(0,0,logo.cols, logo.rows));
	imshow("ROI1", roi1);

	//0.9 - roi1在新的图片的占比
	//0.1 - logo在新的图片的占比
	addWeighted(roi1, 0.9, logo, 0.1, 0, roi1);//将roi1与logo区域重叠,形成新的roi1图片,roi1与logo必须是相同尺寸，相同通道的图像阵列
	//namedWindow("原图加logo", WINDOW_NORMAL);
	imshow("原图加logo", srcImg);

	//注意！！！！！！！！！！！！！！！！！

/*
	Mat对象在使用赋值函数或者拷贝函数时，
	因为Mat对象由Mat头部及矩阵值组成,生成的新的Mat对象具有新的头部地址，矩阵值是一个指针，
	在指向拷贝函数时，传递过来的是一个指针,当新的Mat对象值发生改变，
	将改变所有的Mat值,因为矩阵的值只有一份
	因此，在执行addWeighted函数时，新生成的图片保存到对象roi1，
	实际上也改变了srcImg的值，因为两者指向同一个矩阵

	仅仅当使用clone函数或者copyTo函数时，才会生成新的矩阵(图片)
*/

	waitKey(0);

	return 0;
}
