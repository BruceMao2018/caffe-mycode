#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if( argc != 3 ) { cout << "parameter error" << endl; return -1; }
	Mat img = imread(argv[1]);
	if( !img.data ) { cout << "read img error" << endl; return -1; }
	Mat templ = imread(argv[2]);
	if( !templ.data ) { cout << "read templ img error" << endl; return -1; }

	imshow("源图", img);
	imshow("模板图", templ);
	
	Mat result;
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	//result.create(result_rows, result_cols, CV_32FC1); //result 必须是单通道32位浮点型图像，本行可注释，因为后续matchTemplate函数自动将reslut设置成此类型

	matchTemplate(img, templ, result, CV_TM_SQDIFF_NORMED); ///这里我们使用的匹配算法是标准平方差匹配 method=CV_TM_SQDIFF_NORMED，数值越小匹配度越好

	imshow("匹配结果", result);//此处result是一个一维向量，保存的是上面match的值，以图像的形式输出并未多大意义

	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	imshow("归一化后图", result);

	double minVal = -1;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;
	cout << "匹配度: " << minVal << endl;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	cout << "匹配度: " << minVal << endl;
	matchLoc = minLoc;
	rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
	imshow("效果图", img);

	cout << "srcImage.width: " << img.size().width << " srcImage.height: " << img.size().height << " templ.width: " << templ.size().width << " templ.height: " << templ.size().height << " result.width: " << result.size().width << " result.height: " << result.size().height << endl;
	
	waitKey(0);
	return 0;
}
