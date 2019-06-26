#include "mycode.hpp"

int main(int argc, char **argv)
{
	Mat src = imread("fp.png", IMREAD_COLOR);
	if( !src.data ) { cout << "read img error" << endl; return -1; }

	imshow("原始图片", src);

	Mat grayImage = Mat::zeros(src.size(), CV_8UC1);
	cvtColor(src, grayImage, COLOR_BGR2GRAY);
	imshow("灰度图", grayImage);

	int th = 100;
	Mat img2;
	threshold(grayImage, img2, th, 255, CV_THRESH_BINARY);
	imshow("二值化后图", img2);

	vector<Mat> channels;
	split(src, channels);
	imshow("Blue channel", channels[0]);
	imshow("Green channel", channels[1]);
	imshow("Red channel", channels[2]);
	

	threshold(channels[2], channels[2], th, 255, CV_THRESH_BINARY);
	imshow("红色通道二值化", channels[2]);

	Mat img3, img4, img5;
	src.copyTo(img3, channels[0]);
	src.copyTo(img4, channels[1]);
	src.copyTo(img5, channels[2]);

	//imshow("最终图片0", img3);
	//imshow("最终图片1", img4);
	//imshow("最终图片2", img5);

	waitKey(0);
	return 0;
}
