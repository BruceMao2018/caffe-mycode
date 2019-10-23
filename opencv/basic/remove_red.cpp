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

/*
	Mat red_img = Mat::zeros(src.size(), CV_8UC1);
	red_img.setTo(255);
	//Mat red_img = channels[2];
	//Mat red_img = src;
	for( int i = 0; i < red_img.rows; i++)
	{
		//uchar *ptr = red_img.ptr<uchar>(i);
		for( int j = 0; j < red_img.cols; j++)
		{
			cout << red_img.at<uchar>(i, j);
			//cout << " " << red_img.at<Vec3b>(i, j)[1];
			//cout << " " << red_img.at<Vec3b>(i, j)[2];
		}
		cout << endl;
	}
	//imshow("red_img", red_img);
*/

	Mat img3, img4, img5;
	src.copyTo(img3, channels[0]);
	src.copyTo(img4, channels[1]);
	src.copyTo(img5, channels[2]);


	imshow("Blue+Bin", img3);
	imshow("Green+Bin", img4);
	imshow("Red+Bin", img5);

	waitKey(0);
	return 0;
}
