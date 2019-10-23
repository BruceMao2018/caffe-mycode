#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat addSaltNoise(const Mat src, int len);
Mat addGaussNoise(const Mat src);

int main(int argc, char **argv)
{
	srand((int)time(0));//产生随机种子，否则rand()在程序每次运行时的值都与上一次一样

	if(argc != 2) { cout << "parameter error" << endl; return -1; }
	Mat  srcImg = imread(argv[1]);
	if(!srcImg.data) { cout << "read image error" << endl; return -1; }
	imshow("原图", srcImg);

	Mat saltImg = addSaltNoise(srcImg, 3000);
	imshow("加入椒盐噪声后", saltImg);
	Mat gaussImg = addGaussNoise(srcImg);
	//imshow("加入高斯噪声后", gauImg);


	waitKey(0);
	return 0;
}

Mat addSaltNoise(const Mat src, int len)
{
	Mat img = src.clone();

	bool isGray = src.channels()==1?true:false;


 	//椒噪声
	for( int i = 0; i < len; i++)
	{
		int row = rand()%src.rows;
		int col = rand()%src.cols;

		if( i%500 == 0) cout << "row: " << row << "	col: " << col << endl;

		if( isGray )
			img.at<uchar>(row, col) = 0;
		else
			img.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
			//img.at<Vec3b>(row, col)[0] = 255;
			//img.at<Vec3b>(row, col)[1] = 255;
			//img.at<Vec3b>(row, col)[2] = 255;
	}

	//盐噪声
	for( int i = 0; i < len; i++)
	{
		int row = rand()%src.rows;
		int col = rand()%src.cols;

		if( i%100 == 0) cout << "row: " << row << "	col: " << col << endl;

		if( isGray )
			img.at<uchar>(row, col) = 255;
		else
			img.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
	}

	return img;
}

Mat addGaussNoise(const Mat)
{
	Mat img;
	return img;
}
