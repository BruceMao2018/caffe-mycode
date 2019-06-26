#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	cv::Mat m1(300,300,CV_8UC3);

	Mat logo = imread("logo2.bmp");

	for( int i = 100; i < 400; i++)
		for( int j = 100; j < 400; j++)
			m1.at<Vec3b>(i-100,j-100) = logo.at<Vec3b>(i,j);
		

	cv::imshow("newlogo", m1);
	cv::imwrite("logo3.bmp", m1);

	cv::waitKey(0);

	return 0;
}
