#include "Platform.h"

#ifdef WIN32

BOOL CreateThread(HANDLE& hThreadId, LPTHREAD_START_ROUTINE fun, LPVOID pParam)
{
	hThreadId = CreateThread(NULL, 0, fun, pParam, 0, NULL);
	return (hThreadId != NULL);
}

void WaitForThread(HANDLE hThreadId)
{
	WaitForSingleObject(hThreadId, INFINITE);
}

void TerminateThread(HANDLE hThreadId)
{
	TerminateThread(hThreadId, 0x00);
}

#endif

#ifdef LINUX
BOOL CreateThread(HANDLE& hThreadId, LPTHREAD_START_ROUTINE fun, LPVOID pParam)
{
	return (pthread_create(&hThreadId, NULL, fun, pParam) == 0);
}

void WaitForThread(HANDLE hThreadId)
{
	void *thread_result;
	pthread_join(hThreadId, &thread_result);
}

void TerminateThread(HANDLE hThreadId)
{
	pthread_exit(&hThreadId);
}

void Sleep(int ms)
{
	usleep(ms*1000);
}

HINSTANCE LoadLibrary(const char* filename)
{
	return dlopen(filename, RTLD_NOW);
}

LPVOID GetProcAddress(HINSTANCE hDll, const char* funname)
{
	return dlsym(hDll, funname);
}

BOOL FreeLibrary(HINSTANCE hDll)
{
	return dlclose(hDll);
}

unsigned long GetTickCount()
{  
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((ts.tv_sec*1000) + (ts.tv_nsec/1000000));
}  

BOOL CloseHandle(HANDLE hThreadId)
{
	return TRUE;
}
#endif