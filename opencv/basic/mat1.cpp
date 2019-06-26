#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat src1(3,2,CV_8UC1);

	//src1 = cv::imread(argv[1], IMREAD_COLOR);

	Mat src2(3,2,CV_8UC3, Scalar(0,0,255));
	cout << "src1: " << src1 << endl;
	cout << "src2: " << src2 << endl;

	cout << "src1.at<uchar>(0,0) = " << src1.at<uchar>(0,0) << endl;

	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++)
			src1.at<uchar>(i,j) = (i+j)%255;
		
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++)
			cout << src1.at<uchar>(i,j);

	cout << endl;	

	for (int i = 0; i < src2.rows; i++)
		for (int j = 0; j < src2.cols; j++)
		{
			Vec3b pix;
			pix[0] = 1;
			pix[1] = 2;
			pix[2] = 3;
		
			src2.at<Vec3b>(i,j) = pix;
		}
		
	for (int i = 0; i < src2.rows; i++)
		for (int j = 0; j < src2.cols; j++)
			cout << src2.at<Vec3b>(i,j);
	cout << endl;	

	//显示图像
	imshow("src1: ", src1);
	imshow("src2: ", src2);
	

	return 0;
}
