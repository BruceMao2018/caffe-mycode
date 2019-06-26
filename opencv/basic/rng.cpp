#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	RNG rng1((unsigned)time(NULL));
	for (int i = 0; i < 10; i++)
	{
		int a = rng1.uniform(1, 100);
		cout << a << endl;
	}
	
	return 0;
}

/*
	RNG是opencv中的一个产生伪随机数的类,如果随机种子固定，则每次产生的随机数固定，如果种子变化，则每次随机数变化，实际工作中为了得到不同的随机数，一般使用时间函数作为随机种子来获取随机数
*/
