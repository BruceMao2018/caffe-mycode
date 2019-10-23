#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat addSaltNoise(const Mat srcImage, int n);
Mat addGaussianNoise(const Mat &srcImage);
double generateGaussianNoise(double m, double sigma);

int main(int argc, char **argv)
{

	Mat image = imread("test2.png");
	imshow("原图", image);

	srand((int)time(0));//产生随机种子，否则rand()在程序每次运行时的值都与上一次一样,此srand改变的是整个程序的随机种子，可作用于下面调用的子函数

	Mat dstImage = addSaltNoise(image, 3000);
	imshow("添加椒盐噪声的效果图", dstImage);
	//imwrite("salt_pepper_Image.png", dstImage);

	Mat dstGaussianNoiseImage = addGaussianNoise(image);
	imshow("添加高斯噪声的效果图", dstGaussianNoiseImage);
	imwrite("gaussian_Noice_Image.png", dstGaussianNoiseImage);

	Mat out_box1, out_box2;
	//boxFilter(image, out_box, -1, Size(5,5));
	boxFilter(dstImage, out_box1, -1, Size(5,5));
	boxFilter(dstGaussianNoiseImage, out_box2, -1, Size(5,5));
	imshow("线性方框滤波效果图-salt", out_box1);
	imshow("线性方框滤波效果图-Gaussian", out_box2);

	Mat out_mean1, out_mean2;
	//blur(image, out_mean, Size(5,5));
	blur(dstImage, out_mean1, Size(5,5));
	blur(dstGaussianNoiseImage, out_mean2, Size(5,5));
	imshow("线性均值滤波(低通滤波)效果图-salt", out_mean1);
	imshow("线性均值滤波(低通滤波)效果图-Gaussian", out_mean2);

	Mat out_gaussian1, out_gaussian2;
	//GaussianBlur(image, out_gaussian, Size(5,5),0,0);
	GaussianBlur(dstImage, out_gaussian1, Size(5,5),0,0);
	GaussianBlur(dstGaussianNoiseImage, out_gaussian2, Size(5,5),0,0);
	imshow("线性高斯滤波效果图-salt", out_gaussian1);
	imshow("线性高斯滤波效果图-Gaussian", out_gaussian2);

	Mat out_median1, out_median2;
	//medianBlur(image, out_median, 5);
	medianBlur(dstImage, out_median1, 5);
	medianBlur(dstGaussianNoiseImage, out_median2, 5);
	imshow("非线性中值滤波(中通滤波)效果图-salt", out_median1);
	imshow("非线性中值滤波(中通滤波)效果图-Gaussian", out_median2);

	Mat out_bilateral1, out_bilateral2;
	//bilateralFilter(image, out_bilateral, 25, 25*2, 25/2);
	bilateralFilter(dstImage, out_bilateral1, 25, 25*2, 25/2);
	bilateralFilter(dstGaussianNoiseImage, out_bilateral2, 25, 25*2, 25/2);
	imshow("非线性双边滤波效果图-salt", out_bilateral1);
	imshow("非线性双边滤波效果图-Gaussian", out_bilateral2);


/*
	Mat out_median_salt;
	medianBlur(dstImage, out_median_salt, 5);
	imshow("针对增加了椒盐噪声的图片使用中值滤波器去除椒盐", out_median_salt);
*/
	

	waitKey(0);
	return 0;
}

Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();

	cout << "row: " << dstImage.rows << " cols: " << dstImage.rows << " channels: " << dstImage.channels() << endl;

	//盐噪声
	for( int k = 0; k < n; k++ )
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		if( dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}

 	//椒噪声
	for( int k = 0; k < n; k++ )
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		if( dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}

	return dstImage;
}

double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	
	const double epsilon = numeric_limits<double>::min(); //c++编译器允许的double型的最小值
	static double z0, z1;
	static bool flag = false;
	
	flag = !flag;
	//flag为假构造高斯随机变量X
	if( !flag )
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);//0-1之间随机数(double型)
		u2 = rand() * (1.0 / RAND_MAX);
	}while (u1 <= epsilon );

	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}

Mat addGaussianNoise(Mat &srcImage)
{
	Mat dstImage = srcImage.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//判断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//添加高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}

/*
	去除椒盐噪声使用中值滤波
		1.椒盐噪声

椒盐噪声也称为脉冲噪声，是图像中经常见到的一种噪声，它是一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）。盐和胡椒噪声的成因可能是影像讯号受到突如其来的强烈干扰而产生、类比数位转换器或位元传输错误等。例如失效的感应器导致像素值为最小值，饱和的感应器导致像素值为最大值。

	高斯噪声

高斯噪声是指高绿密度函数服从高斯分布的一类噪声。特别的，如果一个噪声，它的幅度分布服从高斯分布，而它的功率谱密度有事均匀分布的，则称这个噪声为高斯白噪声。高斯白噪声二阶矩不相关，一阶矩为常数，是指先后信号在时间上的相关性。高斯噪声包括热噪声和三里噪声。高斯噪声万有由它的事变平均值和两瞬时的协方差函数来确定，若噪声是平稳的，则平均值与时间无关，而协方差函数则变成仅和所考虑的两瞬时之差有关的相关函数，在意义上它等同于功率谱密度。高斯早生可以用大量独立的脉冲产生，从而在任何有限时间间隔内，这些脉冲中的每一个买充值与所有脉冲值得总和相比都可忽略不计。
*/
