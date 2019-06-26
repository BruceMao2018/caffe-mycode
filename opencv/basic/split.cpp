#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat srcImage;
	Mat imageROI0, imageROI1, imageROI2 ;
	Mat logoImage;

	vector <Mat> channels;
	srcImage = imread("dota.bmp");
	if( !srcImage.data )
	{
		cout << "read img error" << endl;
		return -1;
	}
	imshow("Original-srcImage", srcImage);
	logoImage = imread("logo3.bmp", 0);
	if( !logoImage.data )
	{
		cout << "read logo img error" << endl;
		return -1;
	}

	split(srcImage, channels);//函数原型如下:
	/*
	C++: void split(const Mat& src, Mat*mvbegin);
	C++: void split(InputArray m,OutputArrayOfArrays mv);
	*/
	imageROI0 = channels.at(0);//此at函数是vector的函数，并非OpenCV中Mat类的函数
	imageROI1 = channels[1];
	imageROI2 = channels.at(2);

	cout << "srcImage.rows: " << srcImage.rows << " srcImage.cols: " << srcImage.cols << " srcImage.channels: " << srcImage.channels() << endl;
	cout << "logoImage.rows: " << logoImage.rows << " logoImage.cols: " << logoImage.cols << " logoImage.channels: " << logoImage.channels() << endl;

	addWeighted(imageROI0(Rect(300,100,logoImage.cols, logoImage.rows)),1.0,logoImage,0.5,0.,imageROI0(Rect(300,100,logoImage.cols, logoImage.rows)));

	imshow("srcImageBeforeMerge", srcImage);//如果未对分类后的通道进行merge，则不会作用到原始图片，split函数可视为将所有通道值进行了拷贝，而非单纯的地址拷贝
	merge(channels, srcImage);

	imshow("imageROI0", imageROI0);
	imshow("srcImageAfterMerge", srcImage);

	waitKey(0);


	return 0;
}
