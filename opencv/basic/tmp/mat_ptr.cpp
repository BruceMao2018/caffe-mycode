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
	//Mat img = imread(argv[1], 0);
	if( !img.data ) { cout << "read img error" << endl; return -1;}

	//imshow("original", img);

	cout << "the image info as below: " << endl;
	cout << "Channel: " << img.channels() << " row: " << img.rows << " col: " << img.cols << " size: " << img.size << " dims: " << img.size << endl;

	start = GetTickCount();
	cout << "start: " << start << endl;

//单通道,多通道通用，其原理为，以行为单位，逐个访问每一行的元素，每一行攻击有cols*channels个元素
	int cols3= img.cols * img.channels();//此处为列数乘以通道数, 此种访问方式最快
	for( int i = 0; i < img.rows; i++)//around 27 ms
	{
		uchar *ptr = img.ptr<uchar>(i);//获取第i行的首地址
		for (int j = 0; j < cols3; j++)
		{
			*ptr++ = 0;//逐个访问第i行的每一个元素
			//ptr[j+0] = 0; //错误
			//ptr[j+1] = 0; //错误
			//ptr[j+2] = 0; //错误
		}
	}

	end = GetTickCount();
	cout << "end: " << end << endl;
	cout << "we total spend " << end - start << " mile seconds !" << endl;


	imshow("change1", img);
	waitKey(0);
	return 0;
}
