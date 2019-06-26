#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	if( argc != 2 ) { cout << "parameter error" << endl; return -1; }

	Mat srcImg = imread(argv[1], IMREAD_COLOR);
	if( !srcImg.data ) { cout << "read img error" << endl; return -1; }

	//灰度化
	Mat srcGray, midGray;
	cvtColor(srcImg, srcGray, CV_RGB2GRAY);
	imshow("灰度化后图", srcGray);

	vector<vector<Point>> contours;
	int thre = 50;
	int total = 200;
	int min = 20;
	int max = 40;
while(total--)
{
	contours.clear();
	srcGray.copyTo(midGray);
	//二值化
	threshold(midGray, midGray, thre, 200, CV_THRESH_BINARY);

	//膨胀运算,使得黑白更加分明，以便于查找轮廓
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	//morphologyEx(midGray, midGray, MORPH_DILATE, element);
	//morphologyEx(srcGray, srcGray, MORPH_DILATE, element);

	//寻找轮廓
	//findContours(midGray, contours, RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

	vector<Vec4i> hierarchy;
        findContours(midGray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int sum = contours.size();
	if( sum > max )
		thre--; //当thre值越小，则意味着亮度增加，黑白更加分明,所以当查到的轮廓数量过多，则可调小thre值来获得更好的轮廓图
	else if(contours.size() < min)
		thre++; //查询到的轮廓图也不能太少，否则容易失真
	else
		break;
}
	cout << "loop times: " << 200 - total << endl;
	cout << "final thre: " << thre << endl;
	cout << "total find " << contours.size() << " contours" << endl;
	vector<vector<Point>> contours1;
	imshow("轮廓图", midGray);

	double max_area = 0;
	int max_index = 0;
	for( int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if( area > max_area )
		{
			max_area = area;
			max_index = i;
		}
	}
	cout << "max_index: " << max_index << endl;
	contours1.push_back(contours[max_index]);

	//画出轮廓图,使用白色进行填充
/*
	Mat hole(srcGray.size(), CV_8U, Scalar(0)); //遮罩图层
	drawContours(hole, contours1, -1, Scalar(255), CV_FILLED); 
*/
	drawContours(srcGray, contours1, -1, Scalar(255), CV_FILLED); 
	imshow("轮廓填充图", srcGray);

	Mat RatationedImg(srcImg.size(), CV_8UC1);//旋转后的图片大小与原图一致
	RatationedImg.setTo(0);


drawContours(RatationedImg, contours1, -1, Scalar(255), CV_FILLED);
imshow("旋转矩形标识before", RatationedImg);

	RotatedRect rRect = minAreaRect(contours[max_index]);//最小外接旋转矩形
	double angle = rRect.angle;
	cout << "angle: " << angle << endl;

	Point2f pnt[4];
        rRect.points(pnt); //将旋转矩形的4个点存入到数组
        for (int i = 0; i < 4; i++)
                line(RatationedImg, pnt[i], pnt[(i+1)%4], Scalar(0, 255, 0), 3, 8, 0);

	imshow("旋转矩形标识", RatationedImg);

	//进行仿射变换
	Point2f center = rRect.center;  //中心点
	Mat M2 = getRotationMatrix2D(center, angle, 1);//计算旋转加缩放的变换矩阵
	warpAffine(srcImg, RatationedImg, M2, srcImg.size(),1, 0, Scalar(0));//仿射变换

	imshow("矫正后图片", RatationedImg);

	waitKey(0);
	return 0;
}
