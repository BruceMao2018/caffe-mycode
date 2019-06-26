
#include "Common.h"
#include "OpenCV.h"

int main(int argc, char* argv[])
{
	FILE* fp1 = fopen("D:/DataSet/Test/Seal/New/train.txt", "wt");
	FILE* fp2 = fopen("D:/DataSet/Test/Seal/New/test.txt", "wt");
	FILE* fp3 = fopen("D:/DataSet/Test/Seal/New/list.txt", "wt");

	vector<string> directories;
	directories.push_back("D:/DataSet/Seal/New/Circle");
	directories.push_back("D:/DataSet/Seal/New/Ellipse");
	directories.push_back("D:/DataSet/Seal/New/Square");
	directories.push_back("D:/DataSet/Seal/New/Rectangle");
	directories.push_back("D:/DataSet/Seal/New/Diamond");
	directories.push_back("D:/DataSet/Seal/New/Triangle");

	int nCount = 1;
	RNG rngSet(GetTickCount());
	for (int i=0; i<directories.size(); i++)
	{
		vector<string> files;
		Utility::GetFiles(files, directories[i]);

		for (int j=0; j<files.size(); j++)
		{
			Mat img1 = imread(files[j], IMREAD_COLOR);
			imwrite(Utility::Format("D:/DataSet/Test/Seal/New/%04d-%04d.jpg", i, nCount), img1);

			char filename[128] = {0};
			sprintf(filename, "%04d-%04d.jpg %d\n", i, nCount, i);

			if (rngSet.uniform(0, 1000) <= 800) fputs(filename, fp1);
			else fputs(filename, fp2);
			fputs(filename, fp3);
			
			nCount++;
			printf(filename);
		}
	}

	fclose(fp1);
	fclose(fp2);
	fclose(fp3);

	return 0;
}
