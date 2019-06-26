#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

bool MultiChannelBlending(void) ;

int main(int argc, char **argv)
{
	if( MultiChannelBlending())
	{
		cout << endl << "you got the mixed pic !" << endl;
	}

	waitKey(0);

	return 0;
}


bool MultiChannelBlending()
{
	Mat srcImage;
	Mat logoImage;
	vector <Mat> channels;
	Mat imageBlueChannel;

	srcImage = imread("dota.bmp");
	logoImage = imread("logo3.bmp", 0);//读入灰度图

	if(!srcImage.data) { cout << "read srcImage error" << endl; return false;}
	if(!logoImage.data) { cout << "read logoImage error" << endl; return false;}

	imshow("original-logo", logoImage);

	split(srcImage, channels);
	imageBlueChannel = channels.at(0);

	addWeighted(imageBlueChannel(Rect(300, 100, logoImage.cols, logoImage.rows)), 1.0, logoImage, 0.5, 0, imageBlueChannel(Rect(300, 100, logoImage.cols, logoImage.rows)));

	merge(channels, srcImage);

	imshow("src+logoBlue", srcImage);

	Mat imageGreenChannel;
	//重新读入图片
	srcImage = imread("dota.bmp");
	logoImage = imread("logo3.bmp", 0);//读入灰度图

	if(!srcImage.data) { cout << "read srcImage error" << endl; return false;}
	if(!logoImage.data) { cout << "read logoImage error" << endl; return false;}
	
	split(srcImage, channels);
	imageGreenChannel = channels.at(1);

	addWeighted(imageGreenChannel(Rect(300, 100, logoImage.cols, logoImage.rows)), 1.0, logoImage, 0.5, 0, imageGreenChannel(Rect(300, 100, logoImage.cols, logoImage.rows)));
	merge(channels, srcImage);
	imshow("src+logoGreen", srcImage);
	
	Mat imageRedChannel;
	//重新读入图片
	srcImage = imread("dota.bmp");
	logoImage = imread("logo3.bmp", 0);//读入灰度图

	if(!srcImage.data) { cout << "read srcImage error" << endl; return false;}
	if(!logoImage.data) { cout << "read logoImage error" << endl; return false;}
	
	split(srcImage, channels);
	imageRedChannel = channels.at(2);

	addWeighted(imageRedChannel(Rect(300, 100, logoImage.cols, logoImage.rows)), 1.0, logoImage, 0.5, 0, imageRedChannel(Rect(300, 100, logoImage.cols, logoImage.rows)));
	merge(channels, srcImage);
	imshow("src+logoRed", srcImage);

	return true;
}
