#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "bruceTime.hpp"
#include <unistd.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	unsigned long start, end;

	if(argc != 2) { cout << "parameter error" << endl; return -1;}

	Mat img = imread(argv[1], IMREAD_COLOR);
	//Mat img = imread(argv[1], 0);
	if( !img.data ) { cout << "read img error" << endl; return -1;}

	//imshow("original", img);

	cout << "the image info as below: " << endl;
	cout << "Channel: " << img.channels() << " row: " << img.rows << " col: " << img.cols << " size: " << img.size << " dims: " << img.size << endl;

	start = GetTickCount();
	cout << "start: " << start << endl;

//around 68 ms

//单通道
	if( img.channels() == 1)
	{
		cout << "this is 1 channels image" << endl;
		for( int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				img.at<uchar>(i, j) = 0;
			}
		}
	}
	else
	{
		cout << "this is 3 channels image" << endl;
		for( int i = 0; i < img.rows; i++) //around 69 mile second
		{
			for (int j = 0; j < img.cols; j++)
			{
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 0;

			}
		}
	}
	end = GetTickCount();
	cout << "end: " << end << endl;
	cout << "we total spend " << end - start << " mile seconds !" << endl;

	imshow("change1", img);
	waitKey(0);
	return 0;
}
