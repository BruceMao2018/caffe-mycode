#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat img = imread("test2.png", IMREAD_COLOR);

	Rect rec1 = Rect(100, 300, 600, 200);

	rectangle(img, rec1, Scalar(0, 0, 255), -1, 8, 2);
	
	imshow("rectangle", img);

	waitKey(0);
	return 0;
}
