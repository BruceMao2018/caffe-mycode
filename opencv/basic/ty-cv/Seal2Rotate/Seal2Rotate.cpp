
#include "Common.h"
#include "OpenCV.h"

Mat Preprocess(Mat& img)
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

	vector<Point> points = FindPoint(img3);
	RotatedRect ells = fitEllipse(points);

	Rect box = boundingRect(contours[max_index]);
	//cv::ellipse(img, ells, CV_RGB(0, 0, 255, ), 4, 8);
	//cv::rectangle(img, box, CV_RGB(255, 0, 0, ), 4, 8);

	double angle = ells.angle;

	if (abs((ells.size.width / ells.size.height) - 1.0)<0.18)
	{
		double ratio = (abs(box.x + box.width/2 - ells.center.x) + abs(box.y + box.height/2 - ells.center.y))/ells.size.width;

		if (ratio<0.08)
		{
			Vec4f line;
			fitLine(points, line, CV_DIST_L1, 1, 0.001, 0.001);
			angle = 57.29578049044297*atan2(line[1]/line[0], 1);
		}
		else
		{
			vector<Vec4i> mylines;
			HoughLinesP(img3, mylines, 1, CV_PI/180, 80, 50, 10);

			int min_top = 100000;
			int min_index = -1;
			for (size_t i=0; i<mylines.size(); i++)
			{
				Vec4i line = mylines[i];
				double dis = sqrtf((line[0] - line[2])*(line[0] - line[2]) + (line[1] - line[3])*(line[1] - line[3]));
				if (dis > (box.height+box.width)/4)
				{
					if (min_top > (line[1] + line[3])/2)
					{
						min_top = (line[1] + line[3])/2;
						min_index = i;
						angle = 57.29578049044297*atan2(((float)(line[3] - line[1])) / ((float)(line[2] - line[0])), 1);
					}
				}
			}
		}
	}

	Mat img4 = Rotate(img, angle, CV_RGB(rgb[0], rgb[1], rgb[2]), ells.center);
	Mat img5 = Rotate(img3, angle, CV_RGB(0, 0, 0), ells.center);

	return img4(boundingRect(FindPoint(img5)));
}

int main(int argc, char* argv[])
{
	//vector<string> lines = Utility::GetLines("D:/DataSet/Test/Seal/New2/list.txt");
	vector<string> lines = Utility::GetLines("D:/DataSet/Test/Seal/New/list.txt");

	DWORD start = GetTickCount();

	for (int i=1484; i<lines.size(); i++)
	{
		vector<string> Items = Utility::Split(lines[i], " ");

		string filename = Utility::Format("%04d-%s", i, Items[0].c_str());
/*
		Mat img = imread("D:/DataSet/Test/Seal/New2/" + Items[0], IMREAD_COLOR);
		imwrite("D:/DataSet/Test/Seal/New2/Test/" + filename, Preprocess(img));
*/
///*
		Mat img = imread("D:/DataSet/Test/Seal/New/" + Items[0], IMREAD_COLOR);
		imwrite("D:/DataSet/Test/Seal/New/Test/" + filename, Preprocess(img));
//*/
		printf("%d\n", i);
	}

	printf("%f", (float)(GetTickCount() - start)/lines.size());
	getchar();
	return 0;
}
