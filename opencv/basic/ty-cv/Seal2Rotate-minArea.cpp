
#include "Common.h"
#include "OpenCV.h"

vector<Point> FindPoint(Mat img)
{
	vector<Point> dst;

	for (int i=0; i<img.rows; i++)
		for (int j=0; j<img.cols; j++)
		{
			uchar t = img.at<uchar>(i, j);
			if (t >= 250) dst.push_back(Point(j, i));
		}

	return dst;
}

Mat Preprocess(Mat img)
{
	Mat img2 = Mat::zeros(img.size(), CV_8UC1);

	double rgb[3] = {0.0};
	for (int y=0; y<img.rows; y++)
		for (int x=0; x<img.cols; x++)
		{
			uchar b = img.at<Vec3b>(y, x)[0];
			uchar g = img.at<Vec3b>(y, x)[1];
			uchar r = img.at<Vec3b>(y, x)[2];
			rgb[0] = rgb[0] + r;
			rgb[1] = rgb[1] + g;
			rgb[2] = rgb[2] + b;

			if (r>((g + b)/2)) img2.at<uchar>(y, x) = r - (g + b)/2;
		}
	rgb[0] = rgb[0] / img.total();
	rgb[1] = rgb[1] / img.total();
	rgb[2] = rgb[2] / img.total();

	Mat img3;
	threshold(img2, img3, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	dilate(img3, img2, element);
	dilate(img2, img3, element);
	dilate(img3, img2, element);
	dilate(img2, img3, element);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img3.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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

	img3 = Mat::zeros(img.size(), CV_8UC1);
	cv::drawContours(img3, contours, max_index, Scalar(255), 5);

	RotatedRect rect = cv::minAreaRect(FindPoint(img3));
	double angle = rect.angle;
	if (angle < -50.0) angle = angle + 90;

	Mat img4 = Rotate(img, angle, CV_RGB(rgb[0], rgb[1], rgb[2]), rect.center);
	Mat img5 = Rotate(img3, angle, CV_RGB(0, 0, 0), rect.center);

	return img4(boundingRect(FindPoint(img5)));
}

int main(int argc, char* argv[])
{
	vector<string> lines = Utility::GetLines("D:/DataSet/Test/Seal/New/list.txt");

	for (int i=0; i<lines.size(); i++)
	{
		vector<string> Items = Utility::Split(lines[i], " ");

		imwrite("D:/DataSet/Test/Seal/New/Test/" + Items[0], Preprocess(imread("D:/DataSet/Test/Seal/New/" + Items[0], IMREAD_COLOR)));

		printf("%d\n", i);
	}

	return 0;
}
 