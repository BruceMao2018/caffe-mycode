#include "bubble.hpp"
#include "mycaffe.hpp"
#include <iostream>
#include <unistd.h>
using namespace std;

MyBubble::MyBubble()
{
	cout << "MyBubble construct ..." << endl;
	labelCount = 2;
	CheckedImg = 0;
}

MyBubble::~MyBubble()
{
	cout << "MyBubble Desconstruct ..." << endl;
}

bool MyBubble::LicenseCheck()
{
	sleep(2);
	CheckedImg = 0;

	//system time checking
}

bool MyBubble::Init()
{
	cout << "MyBubble Starting init ..." << endl;
	string deployNet("./bubble_classification_memory.prototxt");
	string caffeModel("./bubble-vgg2_iter_900000.caffemodel");
	MyCaffe = new class MyCaffe(deployNet, caffeModel, labelCount);
	cout << "init done ..." << endl;
}

bool MyBubble::UnInit()
{
	cout << "starting un-init ..." << endl;
	//delete data ;
	//data = NULL;
	//cout << "delete data done" << endl;
	delete MyCaffe;
	MyCaffe = NULL;
	cout << "un-init done ..." << endl;
}

bool MyBubble::BubbleDetect(const char *imgpath, float &v1, float &v2)
{
	if( CheckedImg >= 5 )
	{
		cout << "interface error" << endl;
		v1 = 0;
		v2 = 0;
		return false;
	}

	cout << "imgpath: " << imgpath << endl;

	MyCaffe->ImgDetect(imgpath, v1, v2);

	CheckedImg++;

	return true;
}
