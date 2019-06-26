#include "OpenCV.hh"

void copy(const float* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<float>(i, j) = *(data_src + i*img_dst.rows + j);
}

void copy(const Mat& img_src, float* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<float>(i, j);
}

void copy(const uchar* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<uchar>(i, j) = *(data_src + i*img_dst.rows + j);
}

void copy(const Mat& img_src, uchar* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<uchar>(i, j);
}

void copy(const Mat& src, Mat& dst, const Mat& mask, int offsetx, int offsety)
{
	if (mask.size() != src.size()) return;
	if ((mask.channels() != 1)) return;
	if ((src.channels() != dst.channels())) return;
	if (((src.rows + offsety) > dst.rows)) return;
	if (((src.cols + offsetx) > dst.cols)) return;

	if ((src.channels() == 1) || (src.channels() == 3))
	{
		for (int y=0; y<mask.rows; y++)
			for (int x=0; x<mask.cols; x++)
			{
				uchar ok = mask.at<uchar>(y, x);
				if (ok == 255)
				{
					if (src.channels() == 3)
					{
						dst.at<Vec3b>(offsety + y, offsetx + x) = src.at<Vec3b>(y, x);
					}
					else if (src.channels() == 1)
					{
						dst.at<uchar>(offsety + y, offsetx + x) = src.at<uchar>(y, x);
					}
				}
			}
	}
}

double CosineSimilarity(Mat mVec1, Mat  mVec2)
{
	return (mVec1.dot(mVec2)) / (sqrt(mVec1.dot(mVec1))*sqrt(mVec2.dot(mVec2)));
}

Mat rotate(const Mat& src, double angle, Scalar border, Point2f center)
{
	Mat dst, rot;

	if((center.x == 0.0) || (center.y == 0.0))
		center = Point2f(src.cols/2.0, src.rows/2.0);

	Rect box = RotatedRect(center, src.size(), angle).boundingRect();

	rot = getRotationMatrix2D(center, angle, 1.0);
	rot.at<double>(0, 2) += box.width/2.0 - center.x;
	rot.at<double>(1, 2) += box.height/2.0 - center.y;

	warpAffine(src, dst, rot, box.size(), INTER_LINEAR, 0, border);

	return dst;
}

void printtext(Mat& img, string text, cv::Point center, int font, double scale, Scalar color, int thickness)
{
	int baseline = 0;
	Size size = getTextSize(text, font, scale, thickness, &baseline);

	putText(img, text, Point(center.x - size.width/2, center.y + size.height/2), font, scale, color, thickness);
}
