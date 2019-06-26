#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

int main(int argc, char **argv)
{

	cv::Mat m1(200,200,CV_8UC3);

	cv::MatIterator_<cv::Vec3b> ps, pe;

	ps = m1.begin<cv::Vec3b>();
	pe = m1.end<cv::Vec3b>();

	for( ; ps != pe; ps++)
	{
		(*ps)[0] = 0;
		(*ps)[1] = 0;
		(*ps)[2] = 255;
	}

	cv::imwrite("logo.bmp", m1);

	cv::Mat logo = cv::imread("logo.bmp", cv::IMREAD_COLOR);
	cv::imshow("logo", logo);
	cv::waitKey(0);

	return 0;
}
