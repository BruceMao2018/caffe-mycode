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

	if( img.channels() == 1)
	{
    		/******************单通道的可以这么写***************/
    		//cvtColor(img, img, COLOR_RGB2GRAY); //转化为单通道灰度图

		Mat_<uchar>::iterator it2 = img.begin<uchar>();  //获取起始迭代器
		Mat_<uchar>::iterator it_end2 = img.end<uchar>();  //获取结束迭代器
		for (; it2 != it_end2; it2++)
    		{
            		//在这里分别访问每个通道的元素
            		*it2 = 0;
    		}
	}
	else
	{
		/******************多通道的可以这么写***************/
		Mat_<Vec3b>::iterator it = img.begin<Vec3b>();  //获取起始迭代器
		Mat_<Vec3b>::iterator it_end = img.end<Vec3b>() - img.cols*(img.rows/2);  //获取结束迭代器,迭代器依次从行开始，遍历每一个原始, 如果到第n行结束，则意味着遍历cols*n个元素, 如果到一行就结束，则遍历cols个元素
		for (; it != it_end; it++)
		{
			//在这里分别访问每个通道的元素
			(*it)[0] = 0;
			(*it)[1] = 0;
			(*it)[2] = 0;
		}
	}
	end = GetTickCount();
	cout << "end: " << end << endl;
	cout << "we total spend " << end - start << " mile seconds !" << endl;

	imshow("change1", img);
	waitKey(0);
	return 0;
}
