#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat img = imread(argv[1]);
	if( !img.data ) { cout << "read img error" << endl; return -1; }

	imshow("原始图片", img);

	Mat dst;
	cvtColor(img, img, CV_RGB2GRAY);
	imshow("灰度图", img);
	equalizeHist(img, dst);

	imshow("直方图均衡化", dst);

	waitKey(0);
}
