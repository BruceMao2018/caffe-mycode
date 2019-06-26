
#include "Common.h"
#include "OpenCV.h"

int main(int argc, char* argv[])
{
	FILE* fp1 = fopen("D:/DataSet/Test/Seal/list.txt", "wt");

	Rect r[5] = { Rect(720, 250, 550, 550), Rect(710, 240, 570, 580), Rect(650, 260, 680, 530), Rect(680, 260, 630, 530), Rect(640, 250, 700, 480)};
	
	int nCount = 0;
	for(int i=0; i<5; i++)
		for (int j=0; j<100; j++)
		{
			Mat img1 = imread(Utility::Format("D:/DataSet/Seal/%04d-%04d.png", i+1, j+1), IMREAD_COLOR);
			Mat img2 = img1(r[i]);
			//resize(img1(r[i]), img2, Size(224, 224));
			
			imwrite(Utility::Format("D:/DataSet/Test/Seal/%04d.jpg", nCount), img2);
			//imshow("img2", img2);
			//waitKey();

			fputs(Utility::Format("%04d.jpg %d\n", nCount, nCount).c_str(), fp1);
			printf("%d, %d\n", i, j);

			nCount++;
		}

	fclose(fp1);

	return 0;
}
