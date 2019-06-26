#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat img = imread("test2.png", IMREAD_COLOR);
	if( !img.data ) { cout << "read img error" << endl; return 0; }

	imshow("原始图片", img);
	Rect ccomp;
	floodFill(img, Point(50, 300), Scalar(155,255,255), &ccomp, Scalar(20, 20, 20), Scalar(20, 20, 20));
	imshow("漫水填充效果图", img);

	waitKey(0);
	return 0;
}
