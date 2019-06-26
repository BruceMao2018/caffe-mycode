#include "BaseThread2.h"

BaseThread2::BaseThread2()
{
	hThreadId = NULL;
	bRunning = FALSE;
}

BaseThread2::~BaseThread2()
{
	Stop();
}

BOOL BaseThread2::Start(LPTHREAD_START_ROUTINE pFunc, LPVOID pParam)
{
	if(bRunning == FALSE)
		bRunning = CreateThread(hThreadId, pFunc, pParam);

	return bRunning;
}

BOOL BaseThread2::Stop(BOOL bNow)
{
	if(bRunning)
	{
		bRunning = FALSE;
		if (bNow)
			TerminateThread(hThreadId);
		else
			WaitForThread(hThreadId);		

		return CloseHandle(hThreadId);;
	}
	return TRUE;
}