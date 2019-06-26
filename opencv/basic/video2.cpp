#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char** argv){
    //视频的高度和宽度
    Size s(320,240);
    //创建writer
    VideoWriter writer=VideoWriter("myvideo.avi",CV_FOURCC('M','J','P','G'),25,s);
    //判读创建成功没
    if(!writer.isOpened()){
        cout<<"Can not create video file.\n";
        return -1;
    }
    //视频帧
    Mat frame1(s,CV_8UC3);
    for(int i=0;i<100;i++){
        frame1=Scalar::all(0);
        char text[128];
        snprintf(text,sizeof(text),"%d",i);
        //将数字画到画面上
        putText(frame1,text,Point(s.width/3,s.height/3),
                FONT_HERSHEY_SCRIPT_SIMPLEX,3,
                Scalar(0,0,255),3,8);
        //写入视频中
        writer<<frame1;

    }
  
    //退出时自动关闭
    return 0;
}
