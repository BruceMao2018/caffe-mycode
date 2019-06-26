#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	//定义一个3通道图片，宽度×高度为:200*300,并且使用红色初始化该图片
	Mat img(200,300,CV_8UC3, cv::Scalar(0,0,255));

	//IMREAD_COLOR-表示读取一个彩色,该参数可以省略,如果该参数设置为0，则表示读取一个灰色图片
	//另外,使用imread会改变原图片img的大小
	img = imread("test2.png", IMREAD_COLOR);

	//为防止更改了原始图片，后续所有对图片的操作都使用tmpImg, copyTo()及clone()都可以进行值的拷贝
	Mat tmpImg;
	img.copyTo(tmpImg);

	//定义一个矩型,定义其左上角及矩形的宽度及高度
	int width = 200;
	int heigth = 300;
	Rect rect(100, 200, width, heigth);

	//矩形定义完成，试图从图片上找到矩形定义的位置
	Mat roi = tmpImg(rect);
	

	imshow("原始图片", img);
	imshow("从原始图片找到矩形表示的区域", roi);

	waitKey(0);
	return 0;
}
	
