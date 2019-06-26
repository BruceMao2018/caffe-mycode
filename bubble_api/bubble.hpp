#ifndef BUBBLE_HH
#define BUBBLE_HH
#include "sdk.hpp"
#include "mycaffe.hpp"

class MyBubble : public BubbleSDK
{
public:
	MyBubble();
	virtual ~MyBubble();

	virtual bool Init();
	virtual bool LicenseCheck();
	virtual bool UnInit();
	virtual bool BubbleDetect(const char *imgpath, float &v1, float &v2);

private:
	int CheckedImg;
	int *data ;
	class MyCaffe *MyCaffe;
	int labelCount;
};

#endif
