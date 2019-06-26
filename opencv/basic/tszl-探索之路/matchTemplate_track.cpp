#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat srcImage, templImage, resImage;
int matchMethod = 0;
int maxTrackNum = 50;

void on_matching(int pos, void* data)
{
	cout << "i am on position: " << pos << endl;
	cout << "matchMethod value: " << matchMethod << endl;

	Mat myImage;
	srcImage.copyTo(myImage);
	int res_cols = srcImage.cols - templImage.cols + 1;
	int res_rows = srcImage.rows - templImage.rows + 1;

	resImage.create(res_cols, res_rows, CV_32FC1);
	
	int method = pos/10;
	cout << "with method: " << method << endl;
	matchTemplate(myImage, templImage, resImage, method);
	normalize(resImage, resImage, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(resImage, &minVal, &maxVal, &minLoc, &maxLoc);

	if(method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		cout << "minValue: " << minVal << endl;
	}
	else
	{
		matchLoc = maxLoc;
		cout << "maxValue: " << maxVal << endl;
	}

	rectangle(myImage, matchLoc, Point(matchLoc.x + templImage.cols, matchLoc.y + templImage.rows), Scalar(0, 0, 255), 2, 8, 0);
	
	rectangle(resImage, matchLoc, Point(matchLoc.x + templImage.cols, matchLoc.y + templImage.rows), Scalar(0, 0, 255), 2, 8, 0);

	//imshow("原始图", srcImage);
	imshow("Result", myImage);
}


int main(int argc, char **argv)
{
	if( argc != 3 ) { cout << "parameter error" << endl; return -1; }
	srcImage = imread(argv[1]);
	if( !srcImage.data ) { cout << "read img error" << endl; return -1; }
	templImage = imread(argv[2]);
	if( !templImage.data ) { cout << "read templ img error" << endl; return -1; }

	imshow("源图", srcImage);
	imshow("模板图", templImage);

	namedWindow("Result", WINDOW_AUTOSIZE);
	imshow("Result", srcImage);

	//int maxTrackNum = 5;
	createTrackbar("Method", "Result", &matchMethod, maxTrackNum, on_matching);//地址 &matchMethod 里的int值作为滑轨初始化的值,每次移动滑轨，此地址的值会改变，新的值等于滑轨位置, 同时，这个值也会被回调函数传递(第一个参数, int)
	on_matching(0, NULL);
	
	
	waitKey(0);
	return 0;
}
