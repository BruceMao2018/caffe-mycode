//此程序将实现对图片的整体的随机角度的旋转,并将旋转矩形的外接矩形作为新的图片的大小尺寸
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

	Mat  img = imread("test2.png", IMREAD_COLOR);
	if( !img.data ) { cout << "read img error" << endl; return -1; }

	Point2f centerPoint(img.cols/2, img.rows/2);
	//Size2f mySize(img.cols/2, img.rows/2);
	Size2f mySize = img.size();

	cout << "原始图片size: " << img.size().width << " " << img.size().height << endl;

	RNG rngAngle((unsigned)time(NULL));
	int angle = rngAngle.uniform(0, 360);
	RotatedRect rRect = RotatedRect(centerPoint, mySize, angle);
	Rect box = rRect.boundingRect();

	//获取仿射变换矩阵-2X3的二维矩阵
	//仿射矩阵也可以通过针对3个点的映射关系获得getAffineTransform(srcTriangle, dstTriangle)
	//centerPoint必须是源图片的中心点,不能是外接矩形的中心点
	Mat rot = getRotationMatrix2D(centerPoint, angle, 1.0);

	//对仿射变换矩阵进行改变，以使得仿射后图片的中心点与外接矩形一致
	rot.at<double>(0,2) += box.width/2.0 - centerPoint.x;
	rot.at<double>(1,2) += box.height/2.0 - centerPoint.y;
	warpAffine(img, img, rot, box.size(), INTER_LINEAR, 0, Scalar(128, 128, 128));
	cout << "仿射后图片size: " << img.size().width << " " << img.size().height << endl;
	
/*
	Point2f pnt[4];
	rRect.points(pnt); //将旋转矩形的4个点存入到数组
	for (int i = 0; i < 4; i++)
		line(img, pnt[i], pnt[(i+1)%4], Scalar(0, 255, 0), 3, 8, 0);

	imshow("旋转矩形", img);

	rectangle(img, box, Scalar(255, 0, 0), 5);
	imshow("旋转矩形+外接矩形", img);
*/


	imshow("旋转", img);
	waitKey(0);
	return 0;
}
