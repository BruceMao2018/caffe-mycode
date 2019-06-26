#include "sdk.hpp"
#include "bubble.hpp"
#include <iostream>
using namespace std;

BubbleSDK::BubbleSDK()
{
	cout << "BubbleSDK construct ..." << endl;
}

BubbleSDK::~BubbleSDK()
{
	cout << "BubbleSDK desconstruct ..." << endl;
}

class BubbleSDK *create()
{
	return new MyBubble;
}
