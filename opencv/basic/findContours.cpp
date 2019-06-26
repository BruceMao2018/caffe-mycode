#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat srcImage, grayImage, dstImage;
	srcImage = imread(argv[1]);
	
	if( srcImage.empty() ) { cout << "read img error" << endl; return -1; }

	//namedWindow("原始图像", WINDOW_NORMAL);
	imshow("原始图像", srcImage);

	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	vector <vector<Point>> contours;
	vector <Vec4i> hierarchy;

	grayImage = grayImage > 100;
	findContours(grayImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//findContours(grayImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	dstImage = Mat::zeros(grayImage.size(), CV_8UC3);
	for ( int i = 0; i < hierarchy.size(); i++)
	{
		Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
		drawContours(dstImage, contours, i, color, CV_FILLED, 8, hierarchy);
	}
	
	//namedWindow("轮廓图", WINDOW_NORMAL);
	imshow("轮廓图", dstImage);
	
	waitKey(0);
	return 0;
}
