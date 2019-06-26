#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/*
边缘检测的一般步骤：

    滤波——消除噪声
    增强——使边界轮廓更加明显
    检测——选出边缘点
*/

int main(int argc, char **argv)
{
	Mat im = imread(argv[1], 0);//强制转换成单通道图片
	if( !im.data )
	{
		cout << "read img error" << endl;
		return -1;
	}

	imshow("原始图片灰度图", im);

	//降噪
	Mat edge;
	blur(im, edge, Size(3, 3));

	Mat result;
	//边缘检测
	Canny(edge,result,50,150);

	imshow("边缘检测效果图", result);
	//保存图片
	//imwrite("lena_canny.bmp",result);

	waitKey(0);
	return 0;
}

