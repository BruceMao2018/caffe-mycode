#pragma once

#include "Platform.h"

class MyMutex
{
public:
	MyMutex();
	virtual ~MyMutex();
	
	void Lock();
	void UnLock();

private:
#ifdef WIN32
	HANDLE user_mutexH;                 
#endif
#ifdef LINUX
	pthread_mutex_t user_mutexH; 
#endif
};

//--------------------------------------------------------------------

class MyAutoLock
{
public:
	MyAutoLock(MyMutex* mutex)
	{
		this->mutex = mutex;
		this->mutex->Lock();
	}

	~MyAutoLock()
	{
		mutex->UnLock();
	}

private:
	MyMutex* mutex;
};
