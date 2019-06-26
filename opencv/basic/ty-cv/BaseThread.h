#pragma once

#include "Platform.h"

class BaseThread
{
public:
	BaseThread();
	virtual ~BaseThread();

public:
	BOOL Start();
	BOOL Stop(BOOL bNow = TRUE);
	BOOL GetRunning();
	void SetRunning(BOOL bRunning);

public:
	virtual BOOL Work() = 0;

private:
	HANDLE					hThreadId;
	BOOL					bRunning;
};