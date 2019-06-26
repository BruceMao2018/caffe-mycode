
#include "Common.h"
#include "OpenCV.h"

int main(int argc, char* argv[])
{
	RNG rngAngle(GetTickCount());
	RNG rngIndex(GetTickCount()+10000);
	RNG rngSet(GetTickCount()+20000);

	FILE* fp1 = fopen("D:/DataSet/Test/Seal/New3/train.txt", "wt");
	FILE* fp2 = fopen("D:/DataSet/Test/Seal/New3/test.txt", "wt");
	FILE* fp3 = fopen("D:/DataSet/Test/Seal/New3/list.txt", "wt");
	vector<string> lines = Utility::GetLines("D:/DataSet/Test/Seal/New/list.txt");

	for (int nCount=1; nCount<=40000; nCount++)
	{
		vector<string> Items = Utility::Split(lines[rngIndex.uniform(0, lines.size()-1)], " ");

		Mat img1 = imread("D:/DataSet/Test/Seal/New/" + Items[0], IMREAD_COLOR);
		Mat img2 = Rotate(img1, rngAngle.uniform(0, 360), CV_RGB(128, 128, 128));
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

		Mat img4;
		threshold(img3, img4, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

		Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
		dilate(img4, img3, element);
		dilate(img3, img4, element);
		dilate(img4, img3, element);
		dilate(img3, img4, element);

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

		Rect box = boundingRect(contours[max_index]);
		int width = max(box.width, box.height);
		int height = max(box.width, box.height);

		Mat img5 = Mat(Size(width, height), CV_8UC3, Scalar(rgb[0], rgb[1], rgb[2]));
		Mat imgmask = Mat(Size(box.width, box.height), CV_8UC1, Scalar(255));
		Copy(img2(box), img5, imgmask, (width-box.width)/2, (height-box.height)/2);

		imwrite(Utility::Format("D:/DataSet/Test/Seal/New3/%04d-%04d.jpg", atoi(Items[1].c_str()), nCount), img5);
		string filename = Utility::Format("%04d-%04d.jpg %s\n", atoi(Items[1].c_str()), nCount, Items[1].c_str());

		if (rngSet.uniform(0, 1000) <= 800) fputs(filename.c_str(), fp1);
		else fputs(filename.c_str(), fp2);
		fputs(filename.c_str(), fp3);

		printf("%d, %s", nCount, filename.c_str());
	}

	fclose(fp1);
	fclose(fp2);
	fclose(fp3);

	getchar();
	return 0;
}
