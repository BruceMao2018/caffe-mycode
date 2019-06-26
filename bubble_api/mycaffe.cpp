#include "mycaffe.hpp"
void copy(const float* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<float>(i, j) = *(data_src + i*img_dst.rows + j);
}

void copy(const Mat& img_src, float* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<float>(i, j);
}

void copy(const uchar* data_src, Mat& img_dst)
{
	for (int i=0; i<img_dst.rows; i++)
		for (int j=0; j<img_dst.cols; j++)
			img_dst.at<uchar>(i, j) = *(data_src + i*img_dst.rows + j);
}
void copy(const Mat& img_src, uchar* data_dst)
{
	for (int i=0; i < img_src.rows; i++)
		for (int j=0; j < img_src.cols; j++)
			*(data_dst + i*img_src.rows + j) = img_src.at<uchar>(i, j);
}

float CosineSimilarity(Mat mVec1, Mat  mVec2)
{
	return (mVec1.dot(mVec2)) / (sqrt(mVec1.dot(mVec1))*sqrt(mVec2.dot(mVec2)));
}


MyCaffe::MyCaffe(string v1, string v2, int num)
{
	SetCaffeLog();
	deployNet = v1;
	caffeModel = v2;
	for( int i = 0; i < num; i++)
	{
		labels.push_back(i);
	}
	LoadNet();
}

MyCaffe::~MyCaffe()
{
	cout << "MyCaffe descontract ..." << endl;
	delete net;
	net = NULL;
}


/*
Caffe使用的日志是GLOG，其日志级别如下：
0 - debug
1 - info (still a LOT of outputs)
2 - warnings
3 - errors
*/
void MyCaffe::SetCaffeLog()
{
	//初始化
	google::InitGoogleLogging("xxx");

	//直接关闭日志输出
	//google::ShutdownGoogleLogging();

	//在命令只打印google::ERROR级别以及该级别以上的日志信息
	FLAGS_stderrthreshold = google::ERROR;
	//FLAGS_stderrthreshold = google::INFO;

	//重定位日志信息到特定文件
	//google::SetLogDestination(google::GLOG_FATAL, "log_caffe_err.log");
	//google::SetLogDestination(google::GLOG_ERROR, "log_caffe_err.log");
	//google::SetLogDestination(google::GLOG_WARNING, "log_caffe_err.log");
	//google::SetLogDestination(google::GLOG_INFO, "log_caffe_info.log");
}

void MyCaffe::LoadNet()
{
	Caffe::set_mode(Caffe::CPU);
	net = new caffe::Net<float>(deployNet.c_str(), caffe::TEST);
	net->CopyTrainedLayersFrom(caffeModel.c_str());
}

void MyCaffe::ImgDetect(const char *imgpath, float &v1, float &v2)
{
	vector<Mat> batches;
	Mat src;
	src = cv::imread(imgpath, IMREAD_COLOR);
	cv::resize(src, src, cv::Size(224,224));
	batches.push_back(src);
	MemoryDataLayer<float>* memory_data_layer;
        memory_data_layer = (MemoryDataLayer<float>*)net->layers()[0].get();
	memory_data_layer->AddMatVector(batches, labels);
        net->Forward();
	boost::shared_ptr<caffe::Blob<float> > prob = net->blob_by_name("prob");
        const float* probdata = NULL;
        probdata = prob->cpu_data();
	v1 = probdata[0];
	v2 = probdata[1];

	cout << "v1: " << v1 << " v2: " << v2 << endl;
}
