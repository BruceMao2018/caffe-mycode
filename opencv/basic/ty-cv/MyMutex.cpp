#include "MyMutex.h"

MyMutex::MyMutex()
{
#ifdef WIN32
	user_mutexH = CreateMutex(NULL, FALSE, NULL);
#endif
#ifdef LINUX
	pthread_mutex_init(&user_mutexH, NULL);
#endif
}

MyMutex::~MyMutex()
{
#ifdef WIN32
	CloseHandle(user_mutexH);
#endif
#ifdef LINUX
	pthread_mutex_destroy(&user_mutexH);
#endif
}

void MyMutex::Lock()
{
#ifdef WIN32
	WaitForSingleObject(user_mutexH, INFINITE);
#endif
#ifdef LINUX
	pthread_mutex_lock(&user_mutexH);
#endif
}
	
void MyMutex::UnLock()
{
#ifdef WIN32
	ReleaseMutex(user_mutexH);
#endif
#ifdef LINUX
	pthread_mutex_unlock(&user_mutexH);
#endif
};