#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "bruceTime.hpp"
#include <unistd.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	unsigned long start, end;

	if(argc != 2) { cout << "parameter error" << endl; return -1;}

	Mat img = imread(argv[1], IMREAD_COLOR);
	if( !img.data ) { cout << "read img error" << endl; return -1;}

	//imshow("original", img);

	cout << "the image info as below: " << endl;
	cout << "Channel: " << img.channels() << " row: " << img.rows << " col: " << img.cols << " size: " << img.size << " dims: " << img.size << endl;

	start = GetTickCount();
	cout << "start: " << start << endl;
	for( int i = 0; i < img.rows; i++) //around 29 mile seconds
	{
		for (int j = 0; j < img.cols; j++)
		{
			uchar *p = img.ptr(i, j);
			p[0] = 0;
			p[1] = 0;
			p[2] = 0;
		}
	}
	end = GetTickCount();
	cout << "end: " << end << endl;
	cout << "we total spend " << end - start << " mile seconds !" << endl;


	imshow("change1", img);
	waitKey(0);
	return 0;
}
