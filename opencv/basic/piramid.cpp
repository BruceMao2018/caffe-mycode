#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat srcImage = imread("test2.png");
	Mat tmpImage, dstImage1, dstImage2;
	tmpImage = srcImage;

	imshow("原始图片", tmpImage);

	//进行尺寸调整操作
	resize(tmpImage,dstImage1,Size( tmpImage.cols/2, tmpImage.rows/2 ),(0,0),(0,0),3);
	resize(tmpImage,dstImage2,Size( tmpImage.cols*2, tmpImage.rows*2 ),(0,0),(0,0),3);

	//显示效果图  
	//namedWindow("【效果图】之一", WINDOW_NORMAL);
	//namedWindow("【效果图】之二", WINDOW_NORMAL);
	imshow("【效果图】之一", dstImage1);
	imshow("【效果图】之二", dstImage2);

	Mat dstImage3;
	pyrUp( tmpImage, dstImage3, Size( tmpImage.cols*2, tmpImage.rows*2));
	imshow("pyrUp上取样效果图", dstImage3);
 
	Mat dstImage4;
	pyrDown( tmpImage, dstImage4, Size( tmpImage.cols/2, tmpImage.rows/2));
	imshow("pyrDown下取样效果图", dstImage4);
	waitKey(0);  
	return 0;  
}
