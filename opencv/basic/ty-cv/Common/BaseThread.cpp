#include "BaseThread.h"

THREAD_TYPE BaseThreadFunction(LPVOID pParam)
{
	BaseThread*  Parent = (BaseThread*) pParam;

	while(TRUE)
	{
		if(Parent->GetRunning())
		{
			if(Parent->Work() == FALSE)
				Sleep(20);
		}
		else break;
	}

	return NULL;
}

BaseThread::BaseThread()
{
	hThreadId = NULL;
	bRunning = FALSE;
}

BaseThread::~BaseThread()
{
	Stop();
}

BOOL BaseThread::Start()
{
	if(bRunning == FALSE)
		bRunning = CreateThread(hThreadId, BaseThreadFunction, this);

	return bRunning;
}

BOOL BaseThread::Stop(BOOL bNow)
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

BOOL BaseThread::GetRunning()
{
	return bRunning;
}

void BaseThread::SetRunning(BOOL bRunning)
{
	this->bRunning = bRunning;
}
