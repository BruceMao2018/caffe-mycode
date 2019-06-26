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
	Mat midImage,dstImage;//临时变量和目标图的定义
 
	//【2】进行边缘检测和转化为灰度图
	Canny(srcImage, midImage, 50, 200, 3);//进行一此canny边缘检测
	cvtColor(midImage,dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
 
	//【3】进行霍夫线变换
	vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	HoughLines(midImage, lines, 1, CV_PI/180, 200, 0, 0 );
	//这里注意第五个参数，表示阈值，阈值越大，表明检测的越精准，速度越快，得到的直线越少（得到的直线都是很有把握的直线）
    	//这里得到的lines是包含rho和theta的，而不包括直线上的点，所以下面需要根据得到的rho和theta来建立一条直线
 
	//【4】依次在图中绘制出每条线段
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line( dstImage, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
	}
 
	//【5】显示原始图  
	imshow("【原始图】", srcImage);  
 
	//【6】边缘检测后的图 
	imshow("【边缘检测后的图】", midImage);  
 
	//【7】显示效果图  
	imshow("【效果图】", dstImage);  
 
	waitKey(0);  
 
	return 0;  
}


/*
HoughLinesP(midImage, lines, 1, CV_PI/180, 150, 0, 0 );
void HoughCircles(InputArray image,OutputArray circles, int method, double dp, double minDist, double param1=100,double param2=100, int minRadius=0, int maxRadius=0 )
*/
