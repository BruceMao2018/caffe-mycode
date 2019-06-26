#pragma once

#include "Common.h"
#include "MyMutex.h"

template<class T> class MyList
{
public:
	MyList();
	virtual ~MyList();

public:
	void Push(T it);
	T Pop();

public:
	int Size();
	bool Empty();
	T Front();
	T Back();
	void Clear();

public:
	list<T> all;
	MyMutex* mtx;
};

template<class T> MyList<T>::MyList()
{
	mtx = new MyMutex;
}

template<class T> MyList<T>::~MyList()
{
	Clear();
}

template<class T> void MyList<T>::Push(T it)
{
	MyAutoLock autoLock(mtx);

	all.push_back(it);
}

template<class T> T MyList<T>::Pop()
{
	MyAutoLock autoLock(mtx);

	if(all.size() >= 1)
	{
		T it = all.front();
		all.pop_front(); 
		return it;
	}

	return T();
}

template<class T> int MyList<T>::Size()
{
	MyAutoLock autoLock(mtx);

	return all.size();
}

template<class T> bool MyList<T>::Empty()
{
	MyAutoLock autoLock(mtx);

	return all.empty();
}

template<class T> T MyList<T>::Front()
{
	MyAutoLock autoLock(mtx);

	return all.front();
}

template<class T> T MyList<T>::Back()
{
	MyAutoLock autoLock(mtx);

	return all.back();
}

template<class T> void MyList<T>::Clear()
{
	all.clear();
}
