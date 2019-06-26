#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat m2(600,800,CV_8UC3, Scalar(0,0,0));
	m2 = imread("test1.bmp", cv::IMREAD_COLOR);

	cout << "m2.rows: " << m2.rows << " m2.cols: " << m2.cols << endl;
	//namedWindow("original", WINDOW_NORMAL);
	//imshow("original", m2);

	Mat roi1 = m2(Range(0,50), Range(0,100));

	for (int i = 0; i < roi1.rows; i++)
		for (int j = 0; j < roi1.cols; j++)
		{
			Vec3b pix;
			pix[0] = 0;
			pix[1] = 0;
			pix[2] = 255;
		
			roi1.at<Vec3b>(i,j) = pix;
		}

	imshow("original-red", m2);

	Mat roi2 = m2(Rect(60,40,100,100));//x,y,宽度，高度,注意,此函数的行列数与Range正好相反,x坐标代表的是列，y坐标代表的是行
	cout << "roi2.rows: " << roi2.rows << " roi2.cols: " << roi2.cols << endl;
	imshow("roi2-red", roi2);

	waitKey(0);

	return 0;
}
