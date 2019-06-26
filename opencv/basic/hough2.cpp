#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	//【1】载入原始图和Mat变量定义   
	Mat srcImage = imread(argv[1]);
	if( !srcImage.data ) { cout << "read img error" << endl; return -1; }
	
	Mat midImage;
	cvtColor(srcImage, midImage, CV_BGR2GRAY);

	//降噪
	GaussianBlur(midImage, midImage, Size(9, 9), 2, 2);

	//霍夫圆变换
	vector <Vec3f> circles;
	HoughCircles(midImage, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 200, 0, 0);//注意第七的参数为阈值，可以自行调整，值越大，检测的圆更精准
	
	//依次在图中绘制出圆
	for( size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		//绘制圆心
		circle(srcImage, center, 3, Scalar(0, 255, 0), -1, 8, 0);

		//绘制轮廓
		circle(srcImage, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}

	imshow("效果图", srcImage);

	waitKey(0);
	return 0;
}
