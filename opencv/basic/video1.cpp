#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
    //打开一个视频
    VideoCapture cap("./2.avi");
    //检测是否打开
    if(!cap.isOpened()){
        cout<<"Can not open a camera or file"<<endl;
        return -1;
    }
    
    Mat edges;
    //创建窗口
    namedWindow("edges",1);
    
    for(;;){
        Mat frame;
        //读取一帧
        cap>>frame;
        //判断为空
        if(frame.empty()){
            break;
        }
        //转化为灰度图
        cvtColor(frame,edges,CV_BGR2GRAY);
        //边缘检测
        Canny(edges,edges,0,30,3);
        //显示
        imshow("edges",edges);
        if(waitKey(30)>=0){
            break;
        }
    }
  //退出时自动释放cap
  return 0;
}
