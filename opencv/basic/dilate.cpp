#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat src = imread("test2.png", IMREAD_COLOR);
	imshow("原始图片", src);

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat out;
	dilate(src, out, element);
	imshow("第一次膨胀", out);
	dilate(out, src, element);
	imshow("第二次膨胀", src);
	dilate(src, out, element);
	imshow("第三次膨胀", out);
	dilate(out, src, element);
	imshow("第四次膨胀", src);

	waitKey(0);
	return 0;
}
