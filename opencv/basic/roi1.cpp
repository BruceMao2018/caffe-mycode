#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

int main(int argc, char **argv)
{

	cv::Mat m1(600,600,CV_8UC1);
	cv::Mat firstRow = m1.row(0);
	cv::Mat firstCol = m1.col(0);


	cv::Mat roi1 = m1(cv::Range(1,3), cv::Range(0,2));
	cv::Mat roi2 = m1(cv::Rect(1,0,3,2)); //Rectt四个形参分别是：x坐标，y坐标，长，高；注意(x,y)指的是矩形的左上角点

	cv::Mat m2(100,100,CV_8UC3);
	m2 = imread("test1.bmp", cv::IMREAD_COLOR);//imread后，初始化的图片大小会被改变,图片的大小以打开的图片为准
	if(!m2.data)
	{
		cout << "read image error" << endl;
		return -1;
	}

	cout << "m2.rows: " << m2.rows << "m2.cols: " << m2.cols << endl;
	cv::rectangle(m2, cv::Rect(0,0,100,100), cv::Scalar(0,255,0), 2);
	//cv::imshow("ROI", m2);

	cv::Mat m3 = m2(cv::Rect(0,0,200,200));
	cv::imshow("img-m3", m3);

	cv::Mat m4(m2.rows, m2.cols, CV_8UC3);
	m2(cv::Rect(0,0,400,400)).copyTo(m4);//copyTo函数会改变原图片的大小,完成后新图片的大小取决于copy的尺寸
	cv::imshow("img-m4", m4);


	
	cv::waitKey(0);

	return 0;
}
