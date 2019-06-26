#include "BaseThread3.h"

BaseThread3::BaseThread3(LPTHREAD_START_ROUTINE pFunc, LPVOID pParam)
{
	hThreadId = NULL;
	this->pFunc = pFunc;
	bRunning = FALSE;
	this->pParam = pParam;
}

BaseThread3::~BaseThread3()
{
	Stop();
}

BOOL BaseThread3::Start()
{
	if(bRunning == FALSE)
		bRunning = CreateThread(hThreadId, pFunc, pParam);

	return bRunning;
}

BOOL BaseThread3::Stop(BOOL bNow)
{
	if (bRunning == TRUE)
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

BOOL BaseThread3::GetRunning()
{
	return bRunning;
}
