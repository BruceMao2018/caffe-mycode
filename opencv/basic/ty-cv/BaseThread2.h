#pragma once

#include "Platform.h"

class BaseThread2
{
public:
	BaseThread2();
	virtual ~BaseThread2();

public:
	BOOL Start(LPTHREAD_START_ROUTINE pFunc, LPVOID pParam);
	BOOL Stop(BOOL bNow = TRUE);

private:
	HANDLE					hThreadId;
	BOOL					bRunning;
};