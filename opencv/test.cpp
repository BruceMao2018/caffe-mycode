#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
 

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>

using namespace std;
using namespace cv;
 
int main(int argc,char* argv[])
{

    string modelTxt = "mnist_deploy.prototxt";
    string modelBin = "lenet_iter_10000.caffemodel";
    string imageFile = (argc > 1) ? argv[1] : "5.jpg";

    //! [Create the importer of Caffe model] 导入一个caffe模型接口 
    dnn::Net net = dnn::readNetFromCaffe(modelTxt,modelBin);
  
    if (net.empty()){
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }

    //! [Prepare blob] 读取一张图片并转换到blob数据存储
    Mat img = imread(imageFile,0); //[<Important>] "0" for 1 channel, Mnist accepts 1 channel
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
    // resize(img, img, Size(28, 28));                   //[<Important>]Mnist accepts only 28x28 RGB-images

    // dnn::Blob inputBlob = cv::dnn::Blob(img);   //Convert Mat to dnn::Blob batch of images
    Mat inputBlob = dnn::blobFromImage(img, 1, Size(28, 28));

    Mat prob;
	for(int i = 0; i < 10; i++)
	{
		net.setInput(inputBlob, "data"); 
		prob = net.forward("prob"); 
	}
	Mat probMat = prob.reshape(1, 1); //reshape the blob to 1x1000 matrix // 1000个分类
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber); 
	int classIdx = classNumber.x; // 分类索引号
	printf("n current image classification : %d, possible : %.2f n", classIdx, classProb);
	// putText(img, labels.at(classIdx), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
	// imshow("Image Category", img);
	// waitKey(0);
	return 0;

    //! [Set input blob] 将blob输入到网络
    // net.setInput(inputBlob,"data");        //set the network input

    // //! [Make forward pass] 进行前向传播
    // net.forward();                          //compute output
    // // Mat probMat = probBlob.matRefConst().reshape(1, 1);
    // Mat probMat = prob.reshape(1, 1);
    // //! [Gather output] 获取概率值
    // dnn::Blob prob = net.getBlob("prob");   //[<Important>] gather output of "prob" layer
    // int classId;
    // double classProb;
    // getMaxClass(prob, &classId, &classProb);//find the best class

    // //! [Print results] 输出结果
    // std::cout << "Best class: #" << classId << "'" << std::endl;
    // std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    
    // return 0;
}
