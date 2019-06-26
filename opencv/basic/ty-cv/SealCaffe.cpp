
#include "Common.h"
#include "OpenCV.h"

int main(int argc, char* argv[])
{
	RNG rngAngle(GetTickCount());
	RNG rngLoc(GetTickCount() + 100000);
	RNG rngBg(GetTickCount() + 200000);
	RNG rngSet(GetTickCount() + 400000);

	FILE* fp1 = fopen("D:/DataSet/Test/Seal/Caffe/train.txt", "wt");
	FILE* fp2 = fopen("D:/DataSet/Test/Seal/Caffe/test.txt", "wt"); 
	vector<string> lines = Utility::GetLines("D:/DataSet/Test/Seal/list.txt");

	int nCount = 0;
	for(int i=0; i<lines.size(); i++) 
	{
		vector<string> Items = Utility::Split(lines[i], " ");

		Mat img0 = imread("D:/DataSet/Test/Seal/" + Items[0], IMREAD_COLOR);
		Mat img1 = 255 - img0;

		vector<Mat> channels;
		split(img1, channels);

		Mat img2;
		threshold(channels[0], img2, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(img2.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		int max_index;
		double max_area = 0.0;
		for (int Z=0; Z<contours.size(); Z++)
		{
			double area = contourArea(contours[Z]);
			if (area > max_area)
			{
				max_area = area;
				max_index = Z;
			}
		}

		Mat img3 = Mat::zeros(img1.size(), img1.type());
		cv::drawContours(img3, contours, max_index, CV_RGB(255, 255, 255), CV_FILLED);

	//--------------------------------------------------------------
	//--------------------------------------------------------------
		int nCls = atoi(Items[1].c_str());

		double scale = 1.1;
		for (int X=0; X<80; X++)
		{
			//double angle = rngAngle.uniform(0, 360);
			double angle = 0.0;
			Mat img4 = Rotate(img0, angle, CV_RGB(255, 255, 255));
			Mat img5 = Rotate(img2, angle, CV_RGB(0, 0, 0));
			Mat img6 = Rotate(img3, angle, CV_RGB(0, 0, 0));

			Mat img7;
			cvtColor(img6, img7, CV_BGR2GRAY);

			findContours(img7, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			Rect box = boundingRect(contours[0]);

			Mat imgB;
			do
			{
				int indexB = rngBg.uniform(1, 600);
				//int indexB = 170;
				imgB = imread(Utility::Format("D:/DataSet/Paper/P (%d).jpg", indexB), IMREAD_COLOR);
			} while ((imgB.rows<(box.height*scale)) || (imgB.cols<(box.width*scale)));

			Mat img8 = imgB(Rect(rngLoc.uniform(0, imgB.cols - (int)(box.width*scale)), rngLoc.uniform(0, imgB.rows - (int)(box.height*scale)), (int)(box.width*scale), (int)(box.height*scale)));
			Mat img9 = img4(box);
			Mat img10 = img5(box);
			Copy(img9, img8, img10, box.width*(scale-1.0)/2.0, box.height*(scale-1.0)/2.0);

			nCount++;
			imwrite(Utility::Format("D:/DataSet/Test/Seal/Caffe/%04d-%04d.jpg", nCls, nCount), img8);

			char filename[128] = {0};
			sprintf(filename, "%04d-%04d.jpg %d\n", nCls, nCount, nCls);

			if (rngSet.uniform(0, 1000) <= 800) fputs(filename, fp1);
			else fputs(filename, fp2);
		}
		printf("%d, %s\n", nCls, Items[0].c_str());
	}
	fclose(fp1);
	
	getchar();
	return 0;
}
