#ifndef MYCAFFE_HH__
#define MYCAFFE_HH__

#ifndef USE_OPENCV
#define USE_OPENCV

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "opencv2/opencv.hpp"

#include "caffe/layers/memory_data_layer.hpp"
#include <algorithm>
#include <opencv/highgui.h>


using namespace std;
using namespace cv;
using namespace caffe;
using namespace boost;

class MyCaffe
{
public:
	MyCaffe(string deployNet, string caffeModel, int labelCount);
	~MyCaffe();
	void LoadNet();
	void ImgDetect(const char *imgpath, float &v1, float &v2);
	void SetCaffeLog();
private:
	//enum CaffeMode {Caffe::CPU, Caffe::GPU};
	caffe::Net<float>* net;
	string deployNet;
	string caffeModel;
	vector<int> labels;
};



#endif
#endif
