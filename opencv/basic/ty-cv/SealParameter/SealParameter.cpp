#include "Common.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/world.hpp>

using namespace cv;

#ifdef _DEBUG
#pragma comment(lib, "opencv_world401d.lib")
#endif

#ifdef NDEBUG
#pragma comment(lib, "opencv_world401.lib")
#endif

int array_max(vector<float> data)
{
	float max = -FLT_MAX;
	int index = 0;

	for (int i=0; i<data.size(); i++)
		if (data[i] > max)
		{
			max = data[i];
			index = i;
		}

	return index;
}

void PrintText(Mat& img, string text, Point center, int font, double scale, Scalar color, int thickness)
{
	int baseline = 0;
	Size size = getTextSize(text, font, scale, thickness, &baseline);

	putText(img, text, Point(center.x - size.width / 2, center.y + size.height / 2), font, scale, color, thickness);
}

Mat getSeal(const Mat& img)
{
	Mat img2 = Mat::zeros(img.size(), CV_8UC1);

	double rgb[3] = {0.0};
	for (int y=20; y<img.rows-20; y++)
		for (int x=20; x<img.cols-20; x++)
		{
			uchar b = img.at<Vec3b>(y, x)[0];
			uchar g = img.at<Vec3b>(y, x)[1];
			uchar r = img.at<Vec3b>(y, x)[2];
			if (r>((g+b)/2)) img2.at<uchar>(y, x) = r-(g+b)/2;
		}

	Mat img3;
	blur(img2, img3, Size(20, 20));
	blur(img3, img2, Size(20, 20));

	return img2;
}

void filterContourEllipse(const Mat& img, Mat& imgcontour, vector<Point>& points)
{
	Mat img2;
	threshold(img, img2, 0, 255, THRESH_OTSU | THRESH_BINARY);

	Mat img3;
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	dilate(img2, img3, element);
	dilate(img3, img2, element);
	dilate(img2, img3, element);
	dilate(img3, img2, element);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img2.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	int max_index;
	double max_area = 0.0;
	for (int z=0; z<contours.size(); z++)
	{
		double area = contourArea(contours[z]);
		if (area > max_area)
		{
			max_area = area;
			max_index = z;
		}
	}

	imgcontour = Mat::zeros(img.size(), img.type());
	cv::drawContours(imgcontour, contours, max_index, CV_RGB(255, 255, 255), 2);

	points = contours[max_index];
}

vector<Point> FindPoint(const Mat& img)
{
	vector<Point> dst;

	for (int i=0; i<img.rows; i++)
		for (int j=0; j<img.cols; j++)
		{
			uchar pixel = img.at<uchar>(i, j);
			if (pixel == 255) dst.push_back(Point(j, i));
		}

	return dst;
}

void filterContourTriangle(const Mat& img, Mat& imgcontour, vector<Point>& points)
{
	Mat img2;
	threshold(img, img2, 0, 255, THRESH_OTSU | THRESH_BINARY);
/*
	Mat img3;
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	dilate(img2, img3, element);
	dilate(img3, img2, element);
	dilate(img2, img3, element);
	dilate(img3, img2, element);
*/

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img2.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	for (int z=0; z<contours.size(); z++)
	{
		double area = contourArea(contours[z]);
		if (area > 400)
		{
			for (int i=0; i<contours[z].size(); i++)
				points.push_back(contours[z][i]);
		}
	}
}

RotatedRect getEllipse(const Mat& img)
{
	Mat img2 = getSeal(img);

	vector<Point> points;
	Mat img3;
	filterContourEllipse(img2, img3, points);

	return fitEllipse(points);
}

string getEllipseParameter(Mat& img)
{
	Mat img2;
	resize(img, img2, Size(img.cols*0.85, img.rows*0.85));

	RotatedRect box = getEllipse(img2);
	box.center.x = box.center.x/0.85;
	box.center.y = box.center.y/0.85;

	cv::ellipse(img, box, CV_RGB(0, 0, 255), 4);
	cv::circle(img, box.center, 10, CV_RGB(0, 255, 0), 4);

	return Utility::Format("Center=(%d,%d),Size=(%d,%d),Aangle=%d", (int)(box.center.x), (int)(box.center.y), (int)(box.size.width), (int)(box.size.height), (int)(box.angle));
}

string getCircleParameter(Mat& img)
{
	return getEllipseParameter(img);
}

vector<Point> getVertex(Mat& img)
{
	vector<Point> points;
	Mat img3;
	filterContourTriangle(getSeal(img), img3, points);

	vector<Point> vertex;
	vertex.push_back(Point(img.cols, img.rows));
	vertex.push_back(Point(img.cols, img.rows));
	vertex.push_back(Point(0, 0));
	vertex.push_back(Point(0, 0));
	vertex.push_back(Point(0, 0));

	for (int i=0; i<points.size(); i++)
	{
		if (vertex[0].x > points[i].x) vertex[0] = points[i];
		if (vertex[1].y > points[i].y) vertex[1] = points[i];
		if (vertex[2].x < points[i].x) vertex[2] = points[i];
		if (vertex[3].y < points[i].y) vertex[3] = points[i];
	}
	return vertex;
}

void drawRectangle(Mat& img, RotatedRect box)
{
	vector<Point2f> vt(4);
	box.points(vt.data());

	for (int i = 0; i<vt.size(); i++)
		cv::line(img, vt[i], vt[(i + 1) % 4], CV_RGB(0, 0, 255), 4);
}

string getRectangleParameter(Mat& img)
{
/*
	Mat img2;
	resize(img, img2, Size(img.cols*0.85, img.rows*0.85));

	vector<Point> vertex = getVertex(img2);
	for (int i=0; i<4; i++)
	{
		vertex[i].x = vertex[i].x / 0.85;
		vertex[i].y = vertex[i].y / 0.85;
	}
*/
///*
	vector<Point> vertex = getVertex(img);

	for (int i=0; i<4; i++)
	{
		vertex[4].x = vertex[4].x + vertex[i].x;
		vertex[4].y = vertex[4].y + vertex[i].y;
	}
	vertex[4].x = vertex[4].x/4.0;
	vertex[4].y = vertex[4].y/4.0;

	for (int i=0; i<4; i++)
		cv::line(img, vertex[i], vertex[(i+1)%4], CV_RGB(0, 0, 255), 4);

	cv::circle(img, vertex[4], 10, CV_RGB(0, 255, 0), 4);

	return Utility::Format("Center=(%d,%d),Vertex=((%d,%d),(%d,%d),(%d,%d),(%d,%d))", vertex[4].x, vertex[4].y, vertex[0].x, vertex[0].y, vertex[1].x, vertex[1].y, vertex[2].x, vertex[2].y, vertex[3].x, vertex[3].y);
//*/

	Mat img2;
	resize(img, img2, Size(img.cols*0.85, img.rows*0.85));

	RotatedRect box = getEllipse(img2);
	box.center.x = box.center.x/0.85;
	box.center.y = box.center.y/0.85;
	box.size.height = box.size.height*0.85;
	box.size.width = box.size.width*0.85;

	//cv::ellipse(img, box, CV_RGB(0, 0, 255), 4);
	drawRectangle(img, box);
	cv::circle(img, box.center, 10, CV_RGB(0, 255, 0), 4);

	return Utility::Format("Center=(%d,%d),Size=(%d,%d),Aangle=%d", (int)(box.center.x), (int)(box.center.y), (int)(box.size.width), (int)(box.size.height), (int)(box.angle));
}

string getSquareParameter(Mat& img)
{
	return getRectangleParameter(img);
}

Mat Rotate(const Mat& src, double angle, Scalar border, Point2f center)
{
	if (angle == 0.0) return src.clone();
	Mat dst, rot;

	if ((center.x == 0.0) || (center.y == 0.0))
		center = Point2f(src.cols / 2.0, src.rows / 2.0);

	Rect box = RotatedRect(center, src.size(), angle).boundingRect();

	rot = getRotationMatrix2D(center, angle, 1.0);
	rot.at<double>(0, 2) += box.width / 2.0 - center.x;
	rot.at<double>(1, 2) += box.height / 2.0 - center.y;

	warpAffine(src, dst, rot, box.size(), INTER_LINEAR, 0, border);

	return dst;
}

string getDiamondParameter(Mat& img)
{
	Mat img2;
	resize(img, img2, Size(img.cols*0.85, img.rows*0.85));

	RotatedRect box = getEllipse(img2);
	Mat img3 = Rotate(img, box.angle, CV_RGB(0, 0, 0), cv::Point2f(0, 0));

	vector<Point> vertex = getVertex(img3);

	for (int i=0; i<4; i++)
	{
		vertex[4].x = vertex[4].x + vertex[i].x;
		vertex[4].y = vertex[4].y + vertex[i].y;
	}
	vertex[4].x = vertex[4].x / 4.0;
	vertex[4].y = vertex[4].y / 4.0;

	for (int i=0; i<4; i++)
		cv::line(img3, vertex[i], vertex[(i+1)%4], CV_RGB(0, 0, 255), 4);

	cv::circle(img3, vertex[4], 10, CV_RGB(0, 255, 0), 4);

	img = Rotate(img3, 0-box.angle, CV_RGB(0, 0, 0), cv::Point2f(0, 0));;

	//cv::ellipse(img, box, CV_RGB(0, 0, 255), 4);
	//drawRectangle(img, box);
	//cv::circle(img, box.center, 10, CV_RGB(0, 255, 0), 4);

	return Utility::Format("Center=(%d,%d),Size=(%d,%d),Aangle=%d", (int)(box.center.x), (int)(box.center.y), (int)(box.size.width), (int)(box.size.height), (int)(box.angle));
}

double distance(Point p1, Point p2)
{
	return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
}

string getTriangleParameter(Mat& img)
{
	vector<Point> vertex = getVertex(img);

	double mindis = 10000000;
	int mini, minj;
	for (int i=0; i<4; i++)
	{
		double dis = distance(vertex[i], vertex[(i+1)%4]);
		if (dis < mindis)
		{
			mini = i;
			minj = (i+1)%4;
			mindis = dis;
		}
	}
	vector<Point> vertex2;
	for(int i=0; i<4; i++)
		if((i!=mini) && (i!= minj)) vertex2.push_back(vertex[i]);
	vertex2.push_back(vertex[mini]);

	for (int i=0; i<3; i++)
	{
		vertex[4].x += vertex2[i].x;
		vertex[4].y += vertex2[i].y;
	}
	vertex[4].x = vertex[4].x / 3.0;
	vertex[4].y = vertex[4].y / 3.0;

	for (int i=0; i<3; i++)
		cv::line(img, vertex2[i], vertex2[(i+1)%3], CV_RGB(0, 0, 255), 4);

	cv::circle(img, vertex[4], 10, CV_RGB(0, 255, 0), 4);

	return Utility::Format("Center=(%d,%d),Vertex=((%d,%d),(%d,%d),(%d,%d))", vertex[4].x, vertex[4].y, vertex2[0].x, vertex2[0].y, vertex2[1].x, vertex2[1].y, vertex2[2].x, vertex2[2].y);
}

int main()
{
	Log log;
	log.Init("log.txt", 3);

	String modelTxt = "../../../Train/Caffe/Seal/vggnet7/vggnet_classification_opencv.prototxt";
	String modelBin = "../../../Train/Caffe/Seal/vggnet7/vggnet_iter_20000.caffemodel";

	dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	net.setPreferableTarget(dnn::DNN_TARGET_OPENCL);

	int nCount = 0;
	if (net.empty() == false)
	{
		vector<string> classNames;
		classNames.push_back("Circle");
		classNames.push_back("Ellipse");
		classNames.push_back("Square");
		classNames.push_back("Rectangle");
		classNames.push_back("Diamond");
		classNames.push_back("Triangle");

		vector<string> lines = Utility::GetLines("D:/DataSet/Test/Seal/New3/test.txt");

		DWORD start = GetTickCount();
		for (int X=5414; X<lines.size(); X++)
		{
			vector<string> Items = Utility::Split(lines[X], " ");

			Mat img = imread("D:/DataSet/Test/Seal/New3/" + Items[0], IMREAD_COLOR);

			net.setInput(dnn::blobFromImage(img, 1.0/255.0, Size(224, 224), Scalar(0, 0, 0)), "data");

			string parameter;
			Mat prob = net.forward("prob");
			int classification = array_max(prob);
			switch (classification)
			{
			case 0:
			{
				parameter = getCircleParameter(img);
				break;
			}
			case 1:
			{
				parameter = getEllipseParameter(img);
				break;
			}
			case 2:
			{
				//parameter = getSquareParameter(img);
				break;
			}
			case 3:
			{
				//parameter = getRectangleParameter(img);
				break;
			}
			case 4:
			{
				parameter = getDiamondParameter(img);
				break;
			}
			case 5:
			{
				parameter = getTriangleParameter(img);
				break;
			}
			default:
			{
				break;
			}
			}

			log.Write("%d, %s, %s\n[%s]\n\n", X, Items[0].c_str(), classNames[classification].c_str(), parameter.c_str());
			
			PrintText(img, classNames[classification], Point(img.cols/2, img.rows/4), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 0, 0), 4);
			PrintText(img, parameter, Point(img.cols/2, img.rows/2), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

			imwrite(Utility::Format("D:/DataSet/Test/Seal/New3/Output/%04d-%s", X, Items[0].c_str()), img);
		}
		log.Write("%.4f\n", (float)(GetTickCount()-start)/lines.size());
	}

	log.UnInit();

	getchar();
	return 0;
}