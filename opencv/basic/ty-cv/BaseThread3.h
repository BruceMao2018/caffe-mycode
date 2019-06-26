#pragma once

#include "Platform.h"

class BaseThread3
{
public:
	BaseThread3(LPTHREAD_START_ROUTINE pFunc, LPVOID pParam);
	virtual ~BaseThread3();

	BOOL Start();
	BOOL Stop(BOOL bNow = TRUE);
	BOOL GetRunning();

private:
	HANDLE					hThreadId;
	LPTHREAD_START_ROUTINE	pFunc;
	BOOL					bRunning;
	LPVOID 					pParam;
};