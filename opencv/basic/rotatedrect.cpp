#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat  img = imread("test2.png", IMREAD_COLOR);

	//构建一个旋转矩形,以200,200为中心点,以100,50为宽度及高度, 夹角为30度
	Point2f centerPoint;//定义中心点
	centerPoint.x = 300;
	centerPoint.y = 300;
	Size2f mySize;//定义选择矩形的宽度及高度
	mySize.width = img.cols/4;
	mySize.height = img.rows/4;
	//RotatedRect rRect = RotatedRect(Point2f(200, 200), Size2f(100, 50), -30);
	RotatedRect rRect = RotatedRect(centerPoint, mySize, -30);

	Point2f pnt[4]; //定义4个点的数组，以存放选择矩形的4个点
	rRect.points(pnt); //将矩形的4个点存入到数组

	cout << "p[0]: " << pnt[0].x << " " << pnt[0].y << " p[1]: " << pnt[1].x << " " << pnt[1].y << " p[2]: " << pnt[2].x << " " << pnt[2].y << " p[3]: " << pnt[3].x << " " << pnt[3].y << endl;
	for (int i = 0; i < 4; i++)
		line(img, pnt[i], pnt[(i+1)%4], Scalar(0, 255, 0), 3, 8, 0);

	imshow("旋转矩形", img);

	Rect brect = rRect.boundingRect();
	cout << "brect.area(): " << brect.area() << endl;
	rectangle(img, brect, Scalar(255, 0, 0), 5);

	imshow("旋转矩形+外接矩形", img);

	waitKey(0);
	return 0;
}
