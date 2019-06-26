#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

int main(int argc, char **argv)
{
	cv::Mat src1(3,2,CV_8UC1);
	cv::Mat src2(3,2,CV_8UC3, cv::Scalar(0,0,255));

	cv::MatIterator_<uchar> src1_start, src1_end;
	cv::MatIterator_<cv::Vec3b> src2_start, src2_end;

	for( src1_start = src1.begin<uchar>(), src1_end = src1.end<uchar>(); src1_start != src1_end; src1_start++)
		*src1_start = rand()%255;

	for( src2_start = src2.begin<cv::Vec3b>(), src2_end = src2.end<cv::Vec3b>(); src2_start != src2_end; src2_start++)
	{
//赋值时不能如下, src2_start[0]是一个指针，指向一个向量(1行3列)
//src2_start是一个指向2维向量的指针,2维向量中的值是一个列向量,因此,src2_start[0]指向的是第1行，src2_start[1]指向的是第二行,该行并不存在，我们的值是1行3列
//src2_start[0] = 1; //赋值结果是{1,0,0}
//src2_start[1] = 2; //该行并不存在，赋值错误
//src2_start[2] = 3; //该行并不存在，赋值错误

//可以使用如下方式赋值, src2_start[0]是一个指针，指向一个1行3列向量, [0][x]指向向量的值,x取值范围0,1,2

//src2_start[0][0] = 1;
//src2_start[0][1] = 2;
//src2_start[0][2] = 3;

//下面赋值方式与上面一样，给一个1行3列的向量赋值
//等于src2_start[0] = {1,2,3};//可以使用这种方式赋值，等于将一个列向量赋值给第一行(仅有1行)
//src2_start[0] = {1,2,3};


		//下面赋值方式等同于src2_start[0][0] = , src2_start[0][1] = , src2_start[0][2] = , 
		(*src2_start)[0] = 0;
		(*src2_start)[1] = 0;
		(*src2_start)[2] = 255;

		//src2_start[0][0] = 0;
	}

	for (int i = 0; i < src2.rows; i++)
		for (int j = 0; j < src2.cols; j++)
			cout << src2.at<cv::Vec3b>(i,j);
	cout << endl;	

//也可以使用如下方式赋值
for(int i=0;i<src1.rows;++i){

//p是i行首地址
    uchar* p=src1.ptr<uchar>(i);
    for(int j=0;j<src1.cols;++j){
      //p[j]=(i+j)%255;
      p[j]=1;
    }
  }

	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++)
			cout << src1.at<uchar>(i,j);
	cout << endl;	

  //遍历colorimg
  for(int i=0;i<src2.rows;++i){
    cv::Vec3b * p=src2.ptr<cv::Vec3b>(i);
    for(int j=0;j<src2.cols;++j){
      p[j][0]=1;
      p[j][1]=2;
      p[j][2]=3;
    }
  }

	for (int i = 0; i < src2.rows; i++)
		for (int j = 0; j < src2.cols; j++)
			cout << src2.at<cv::Vec3b>(i,j);
	cout << endl;	

	return 0;
}
