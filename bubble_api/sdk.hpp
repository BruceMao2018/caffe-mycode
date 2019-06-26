#ifndef SDK_BUBBLE_HH
#define SDK_BUBBLE_HH


class BubbleSDK
{
public:
	BubbleSDK();
	virtual ~BubbleSDK() ;

	virtual bool  Init() = 0;
	virtual bool UnInit() =0;
	virtual bool BubbleDetect(const char *imgpath, float &v1, float &v2) =0;
};

class BubbleSDK *create();

#endif
