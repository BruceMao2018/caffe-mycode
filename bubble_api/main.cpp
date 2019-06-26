#include "sdk.hpp"
#include <iostream>
using namespace std;

int main()
{
	class BubbleSDK *sdk = create();
	sdk->Init();
	float a,b;
	for( int i = 0; i < 2; i++)
	{
		if(sdk->BubbleDetect("/home/bruce/local_install/caffe/mycode/bubble_api/1.bmp", a, b))
			cout << "Get Value: " << a << " " << b << endl;
		else
			cout << "api error" << endl;
	}

	getchar();
	sdk->UnInit();
	delete sdk ;
	sdk = NULL;

	return 0;
}
