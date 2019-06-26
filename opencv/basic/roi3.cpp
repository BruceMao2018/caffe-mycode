#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat srcImg = imread("test1.bmp");
	if( !srcImg.data )
	{
		cout << "read img error" << endl;
		return -1;
	}

	//namedWindow("srcImg", WINDOW_NORMAL);
	imshow("原始图片", srcImg);

	//src1 = cv::imread(argv[1], IMREAD_COLOR);
	Mat logo = imread("logo.bmp");
	Mat mask = imread("logo.bmp", 0);//必须是灰度图
	if( !logo.data )
	{
		cout << "read logo error" << endl;
		return -1;
	}
	imshow("logo图", logo);

	Mat roi = srcImg(Rect(0,0,logo.cols, logo.rows));
	imshow("ROI", roi);

	logo.copyTo(roi, mask);//注意两个参数，一个是ROI,一个是掩模, 也可以不使用掩膜,直接用logo.copyTo(roi), 效果一样
	imshow("原图加logo", srcImg);

	waitKey(0);

	return 0;
}
