#include "atrinet.hpp"

#define NetF float
#define DWORD unsigned long //DWORD is used in Windows, we should use unsigned long or double here


/*
namespace caffe
{
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	extern INSTANTIATE_CLASS(ReLULayer);
	extern INSTANTIATE_CLASS(PoolingLayer);
	extern INSTANTIATE_CLASS(LRNLayer);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(MemoryDataLayer);

	REGISTER_LAYER_CLASS(Convolution);
	REGISTER_LAYER_CLASS(ReLU);
	REGISTER_LAYER_CLASS(Pooling);
	REGISTER_LAYER_CLASS(LRN);
	REGISTER_LAYER_CLASS(Softmax);
}
*/

int main(int argc, char* argv[])
{
	Mat f1(1, 1024, CV_32F);
	Mat f2(1, 1024, CV_32F);
	Mat f3(1, 1024, CV_32F);

	if( argc != 5)
	{
		cout << "Parameter error" << endl;
		cout << "should like: atr deployNet caffeModel labeCount imageName" << endl;
		return -1;
	}

	//string basePath("/home/bruce/local_install/caffe/examples/");
	string basePath("");
	string deployNet = basePath + argv[1];
	string caffeModel = basePath + argv[2];
	
	Caffe::set_mode(Caffe::CPU);
	caffe::Net<float>* net(new caffe::Net<float>(deployNet.c_str(), caffe::TEST));
	net->CopyTrainedLayersFrom(caffeModel.c_str());


	vector<int> labels;
	for(int n = 0; n < atoi(argv[3]); n++)
	{
		labels.push_back(n);
	}

	vector<Mat> batches1;
	vector<Mat> batches2;
	vector<Mat> batches3;

	Mat src1, src2, src3;

	string imageName = basePath + argv[4];
	src1 = cv::imread(imageName.c_str(), IMREAD_COLOR);
	cv::resize(src1, src1, cv::Size(224,224));
	batches1.push_back(src1);

/*
	src2 = cv::imread("/home/bruce/local_install/caffe/examples/atrinet/0002-0001.jpg", IMREAD_COLOR);
	cv::resize(src2, src2, cv::Size(1080,1080));
	batches2.push_back(src2);

	src3 = cv::imread("/home/bruce/local_install/caffe/examples/atrinet/0003-0001.jpg", IMREAD_COLOR);
	cv::resize(src3, src3, cv::Size(1080,1080));
	batches3.push_back(src3);
*/

	cout << "push image done" << endl;



	MemoryDataLayer<float>* memory_data_layer;
	memory_data_layer = (MemoryDataLayer<float>*)net->layers()[0].get();

	cout << "attach to net done" << endl;

/*
	boost::shared_ptr<caffe::Blob<float> > ip1 = net->blob_by_name("ip1");
	const float* ip1data = NULL;
	ip1data = ip1->cpu_data();

	cout << "get ip1 cpu data: " << ip1data << endl;
*/


	DWORD start = getTickCount();

	memory_data_layer->AddMatVector(batches1, labels);
	net->Forward();
	cout << "batches1 forward done" << endl;

	DWORD end = getTickCount();
	cout << "time length for forward: " << end - start << endl;

	boost::shared_ptr<caffe::Blob<float> > prob = net->blob_by_name("prob");
	const float* probdata = NULL;
	probdata = prob->cpu_data();

	cout << "get prob layer cpu data: " << probdata << endl;

	copy(probdata, f1);

	for (int X=0; X<atoi(argv[3]); X++)
		cout << "label " << X << ": " << probdata[X] << endl;

/*
	memory_data_layer->AddMatVector(batches2, labels);
	net->Forward();
	cout << "batches2 forward done" << endl;

	prob = net->blob_by_name("prob");
	probdata = prob->cpu_data();
	copy(probdata, f2);

	prob = net->blob_by_name("prob");
	probdata = prob->cpu_data();
	for (int X=0; X<3; X++)
		cout << "label " << X << ": " << probdata[X] << endl;

	memory_data_layer->AddMatVector(batches3, labels);
	net->Forward();
	cout << "batches3 forward done" << endl;

	prob = net->blob_by_name("prob");
	probdata = prob->cpu_data();
	copy(probdata, f3);

	prob = net->blob_by_name("prob");
	probdata = prob->cpu_data();
	for (int X=0; X<3; X++)
		cout << "label " << X << ": " << probdata[X] << endl;

	printf("%f\n", pow(CosineSimilarity(f1, f1), 2));
	printf("%f\n", pow(CosineSimilarity(f1, f2), 2));
	printf("%f\n", pow(CosineSimilarity(f1, f3), 2));
*/

	//getchar();
	return 0;
}
