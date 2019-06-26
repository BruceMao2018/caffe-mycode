#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	//Mat src = imread("test2.png", IMREAD_COLOR);
	Mat src = imread("test2.png", 0);
	if( !src.data ) { cout << "read img error" << endl; return -1; }
	imshow("原始图片", src);

	Mat dst;
	threshold(src, dst, 0, 255, THRESH_OTSU | THRESH_BINARY );

	imshow("图像二值化效果图", dst);

	waitKey(0);
	return 0;
}
