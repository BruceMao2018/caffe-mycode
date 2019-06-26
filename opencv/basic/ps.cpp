//此程序可以成功提取圆形，椭圆，三角，正方形

#include "mycode.hpp"

int main(int argc, char **argv)
{
	Mat src = imread(argv[1]);
	if( !src.data ) { cout << "read img error" << endl; return -1; }

	Mat gray = Mat::zeros(src.size(), CV_8UC1);
	cvtColor(src, gray, CV_BGR2GRAY);

	imshow("灰度图", gray);

	//二值化
	int th = (100 / 4)*2 + 1;
	Mat img2 = Mat::zeros(src.size(), CV_8UC1);
	//threshold(gray, img2, th, 255, CV_THRESH_BINARY);
	adaptiveThreshold(gray, img2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, th, th/3);
	imshow("二值化图", img2);

	//开运算去噪点
	Mat img3;
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(img2, img3, MORPH_OPEN, element);
	//morphologyEx(img3, img3, MORPH_OPEN, element);
	//morphologyEx(img3, img3, MORPH_OPEN, element);
	//morphologyEx(img3, img3, MORPH_OPEN, element);

	imshow("膨胀后的图片", img3);
	//threshold(img3, img3, th, 255, CV_THRESH_BINARY);
	//imshow("膨胀后二值化的图片", img3);

	vector<vector<Point>> contours, contours_tmp;
	vector<Vec4i> hierarchy;
	findContours(img3, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for( int i = 0; i < hierarchy.size(); i++)
	{
		if(contourArea(contours[i]) > 2000)
		//if (contours[i].size() > 200)//将比较小的轮廓剔除掉
			contours_tmp.push_back(contours[i]);
	}

	Mat hole(gray.size(), CV_8UC1, Scalar(0));
	drawContours(hole, contours_tmp, -1, Scalar(255), CV_FILLED);
	imshow("hole", hole);

	Mat img4;
	src.copyTo(img4, hole);

	imshow("最终图片", img4);

	waitKey(0);
	return 0;
}
