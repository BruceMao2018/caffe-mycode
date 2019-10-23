#include <vector>
#include <iostream>
#include <unistd.h>
#include <list>
#include "bruceTime.hpp"

using namespace std;

int main(int argc, char **argv)
{

	vector <int> ve1;
	list <int> lt1;
	int num = 100000;
	while(num--)
	{
		ve1.push_back(num);
		lt1.push_back(num);
	}

	unsigned long start = GetTickCount();
	cout << "ve1[5000]: " << ve1[5000] << endl;
	unsigned long end = GetTickCount();
	cout << "end - start : " << end - start << endl;

	list <int>::iterator p = lt1.begin();
	start = GetTickCount();
	cout << "lt1[5000]: " << p << endl;
	end = GetTickCount();
	cout << "end - start : " << end - start << endl;

	return 0;
}
