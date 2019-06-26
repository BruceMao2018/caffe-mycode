#include "mycode.hpp"

void Copy(const float* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<float>(i, j) = *(data_src + i*img_dst.rows + j);
}

void Copy(const Mat& img_src, float* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<float>(i, j);
}

void Copy(const uchar* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<uchar>(i, j) = *(data_src + i*img_dst.rows + j);
}

void Copy(const Mat& img_src, uchar* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<uchar>(i, j);
}

void Copy(const Mat& src, Mat& dst, const Mat& mask, int offsetx, int offsety)
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
						dst.at<Vec3b>(offsety+y, offsetx+x) = src.at<Vec3b>(y, x);
					}
					else if (src.channels() == 1)
					{
						dst.at<uchar>(offsety+y, offsetx+x) = src.at<uchar>(y, x);
					}
				}
			}
	}
}

double CosineSimilarity(Mat mVec1, Mat  mVec2)
{
	return (mVec1.dot(mVec2)) / (sqrt(mVec1.dot(mVec1))*sqrt(mVec2.dot(mVec2)));
}
