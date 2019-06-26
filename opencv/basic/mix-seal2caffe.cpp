#include "mycode.hpp"

int main(int argc, char **argv)
{
	Mat src = imread(argv[1], IMREAD_COLOR);
	if( !src.data ) { cout << "read img error" << endl; return -1; }
	imshow("原始图片", src);

	//从源图的中心点进行旋转,然后得到外接矩形图像
	Point2f center;
	center.x = src.cols/2;
	center.y = src.rows/2;

	//旋转角度随机
	RNG rngAngle((unsigned)time(NULL));
	int angle = rngAngle.uniform(0, 360);

	//旋转矩形的大小
	Size2f size;
	size.width = src.cols;
	size.height = src.rows;

	//得到旋转矩形brect
	RotatedRect brect = RotatedRect(center, size, angle);

	//得到旋转矩形的外接矩形
	Rect box = brect.boundingRect();

	//进行图片仿射,仿射的中心点为源图片的中心，角度与旋转角度一样，大小与旋转矩形的外接矩形一致，即或者外接矩形区域的图片
	Scalar border = CV_RGB(128, 128, 128);
	Mat img2;
	Mat rot = getRotationMatrix2D(center, angle, 1.0);

	//旋转后的外接矩形大小与原始图片大小不一致，调整rot参数以使得新的图片完全包含了源图片所有像素
	rot.at<double>(0,2) += box.width/2.0 - center.x;
	rot.at<double>(1,2) += box.height/2.0 - center.y;
	warpAffine(src, img2, rot, box.size(), INTER_LINEAR, 0, border);
	
	imshow("按照外接矩形的角度及尺寸进行仿射后的效果图", img2);

	//img3
	//?????????????????????????
	Mat img3 = Mat::zeros(img2.size(), CV_8UC1);
	double rgb[3] = {0.0};
                for (int y=0; y<img2.rows; y++)
                        for (int x=0; x<img2.cols; x++)
                        {
                                uchar b = img2.at<Vec3b>(y, x)[0];
                                uchar g = img2.at<Vec3b>(y, x)[1];
                                uchar r = img2.at<Vec3b>(y, x)[2]; 
                                rgb[0] = rgb[0] + r;
                                rgb[1] = rgb[1] + g;
                                rgb[2] = rgb[2] + b;

                                if (r>((g + b)/2)) img3.at<uchar>(y, x) = r - (g + b)/2;
                        }
                rgb[0] = rgb[0] / img2.total();
                rgb[1] = rgb[1] / img2.total();
                rgb[2] = rgb[2] / img2.total();

	imshow("经过神秘更改后的img3", img3);

	//img4 -- 将img3进行二值化得到img4
	Mat img4;
	threshold(img3, img4,  0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
	imshow("对img3进行二值化后得到的img4", img4);

	//将二值化的img4进行4次膨胀得到新的img4
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        dilate(img4, img3, element);
        dilate(img3, img4, element);
        dilate(img4, img3, element);
        dilate(img3, img4, element);
	
	imshow("进行了4次膨胀后的img4", img4);

	//查找灰色图img3的所有轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img3, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//查找面积最大的轮廓
	int max_index;
	double max_area = 0.0;
	for( int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if(area > max_area)
		{
			max_area = area;
			max_index = i;
		}
	}
	//获取最大轮廓的外接矩形
	Rect box2 = boundingRect(contours[max_index]);
	int width = max(box2.width, box2.height);
	int height = max(box2.width, box2.height);

	Mat img5 = Mat(Size(width, height), CV_8UC3, Scalar(rgb[0], rgb[1], rgb[2]));
	imshow("最大轮廓外接矩形使用rgb填充后的彩色图片", img5);
	Mat imgmask = Mat(Size(box2.width, box2.height), CV_8UC1, Scalar(255));
	Copy(img2(box2), img5, imgmask, (width-box2.width)/2, (height-box2.height)/2);

	imshow("最终图片", img5);

	waitKey(0);
	return 0;
}
