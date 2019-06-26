#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat image = imread("test2.png");
	if( !image.data ) {cout << "read image error" << endl; return -1; }

	imshow("原始图片", image);
	
	Mat element1 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat element3 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat out1, out2, out3;
	//膨胀
	dilate(image, out1, element1);
	imshow("膨胀效果图", out1);
	
	//腐蚀
	erode(image, out2, element2);
	imshow("腐蚀效果图1", out2);

	//通过调用morphologyEx函数进行腐蚀操作,效果相同
	morphologyEx(image, out2, MORPH_ERODE, element2);
	imshow("腐蚀效果图2", out2);

	erode(out1, out3, element3);
	imshow("对进行过膨胀的图片做腐蚀效果图", out3);

	Mat out4, out5;
	Mat element4 = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(image, out4, MORPH_OPEN, element4);
	morphologyEx(image, out5, MORPH_CLOSE, element4);
	imshow("开运算效果图", out4);
	imshow("闭运算效果图", out5);

	Mat out6, out7;
	morphologyEx(image, out6, MORPH_TOPHAT, element4);
	morphologyEx(image, out7, MORPH_BLACKHAT, element4);
	imshow("顶冒效果图", out6);
	imshow("黑冒效果图", out7);

	Mat out8;
	morphologyEx(image, out8, MORPH_GRADIENT, element4);
	imshow("形态学梯度效果图", out8);

	waitKey(0);
	return 0;
}

/*
	开运算（Opening Operation），其实就是先腐蚀后膨胀的过程。
		开运算可以用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积。
	
	闭运算(Closing Operation), 先膨胀后腐蚀的过程称为闭运算
		闭运算能够排除小型黑洞(黑色区域)

	顶帽运算（Top Hat）又常常被译为”礼帽“运算。为原图像与上文刚刚介绍的“开运算“的结果图之差
		因为开运算带来的结果是放大了裂缝或者局部低亮度的区域，因此，从原图中减去开运算后的图，得到的效果图突出了比原图轮廓周围的区域更明亮的区域，且这一操作和选择的核的大小相关。
顶帽运算往往用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取。

	黑帽（Black Hat）运算为”闭运算“的结果图与原图像之差
		黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，且这一操作和选择的核的大小相关。 所以，黑帽运算用来分离比邻近点暗一些的斑块

*/
