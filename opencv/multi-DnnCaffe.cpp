#ifndef USE_OPENCV
#define USE_OPENCV

#include "opencv2/opencv.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <utility>

using namespace cv;
using namespace cv::dnn;
using namespace std;

/*
#ifdef _DEBUG
#pragma comment(lib, "opencv_world401d.lib")
#endif

#ifdef NDEBUG
#pragma comment(lib, "opencv_world401.lib")
#endif

*/

#define DWORD long int

int array_max(vector <float> data)
{
	float max = -FLT_MAX;
	int index = 0;

	for (int i = 0; i<data.size(); i++)
		if (data[i] > max)
		{
			max = data[i];
			index = i;
		}
	cout << "index: " << index << " max value: " << data[index] << endl;

	return index;
}

void printtext(Mat& img, string text, Point center, int font, double scale, Scalar color, int thickness)
{
	int baseline = 0;
	Size size = getTextSize(text, font, scale, thickness, &baseline);

	putText(img, text, Point(center.x - size.width/2, center.y + size.height/2), font, scale, color, thickness);
}

vector<string> GetLinesAsVector(const char *filename)
{
	vector <string> lines;
	fstream fin;
	fin.open(filename);
	if( !fin)
	{
		cout << "open file error" << endl;
		fin.close();
		exit(-1) ;
	}
	string str;
	while (!fin.eof())
	{
		getline(fin, str);
		if( str.size() > 0 )
			lines.push_back(str);
	}
	fin.close();

	return lines;

}

int main(int argc, char **argv)
{

	cv::Mat src1, src2;
	src1 = cv::imread(argv[1], IMREAD_COLOR);
	src2 = cv::imread(argv[2], IMREAD_COLOR);

	string modelTxt = "/home/bruce/local_install/caffe/examples/bubble/bubble_lenet_classification_input.prototxt";
	string modelBin = "/home/bruce/local_install/caffe/output/bubble/bubble-lenet_iter_5000.caffemodel";

	dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	net.setPreferableTarget(dnn::DNN_TARGET_OPENCL);

	//net.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);

/*
enum  	cv::dnn::Backend {
  cv::dnn::DNN_BACKEND_DEFAULT,
  cv::dnn::DNN_BACKEND_HALIDE,
  cv::dnn::DNN_BACKEND_INFERENCE_ENGINE,
  cv::dnn::DNN_BACKEND_OPENCV,
  cv::dnn::DNN_BACKEND_VKCOM
}*/
	//net.setPreferableTarget(dnn::DNN_TARGET_CPU);

	int nCount = 0;
	if (net.empty() == false)
	{
		vector<string> lines = GetLinesAsVector("/home/bruce/local_install/caffe/examples/bubble/sample.txt");
		vector<string> classNames = {"0", "1"};

		DWORD start = getTickCount();
		int num = lines.size();
		cout << "image num: " << num << endl;
		for (int i=0; i<num; i++)
		{
			Mat img = imread(lines[i], IMREAD_COLOR);
			if (img.empty())
			{
				cout << "read image error" << endl;
				exit(-1);
			}
			//net.setInput(dnn::blobFromImage(img, 1.0/255.0, Size(224, 224), Scalar(0, 0, 0)), "data");

			Mat inputBlob = dnn::blobFromImage(img, 1.0/255.0, Size(224,224), Scalar(0,0,0), false);
			net.setInput( inputBlob, "data"); //set the network input


//fetch data from multi layer
			vector <vector<Mat>> outputBlobs;
			vector <cv::String> outBlobNames;

			outputBlobs.clear();
			//outBlobNames.push_back("ip1");
			//outBlobNames.push_back("ip2");
			outBlobNames.push_back("prob");


			net.forward(outputBlobs, outBlobNames);

			getchar();
			Mat prob = outputBlobs[0][0];

			getchar();
			int test = array_max(prob);
			getchar();
			//int real = atoi(items[1].c_str());
			
			printtext(img, classNames[test], Point(img.cols/2, img.rows/2), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 0, 0), 4);

			//imwrite(items[0], img);
			//printf("%s\n", lines[i].c_str());
		}
		printf("%.4f, %.4f\n", (float)(getTickCount()-start)/lines.size(), ((float)nCount)/lines.size()); //2080ti(42.3500, 13.1300), 1060(161.6844, 52.6600)
	}
	else
	{
		cout << "read net error" << endl;
		exit(-1);
	}

	getchar();
	return 0;
}
#endif
