#pragma once

#include "Platform.h"
#include "MyMutex.h"

class Log
{
public:
	Log();
	virtual ~Log();

public:
	BOOL Init(string strFileName, int nLevel);
	void UnInit();
	void Write(const char* szFormat, ...);
	void Write_noaffix(const char* szFormat, ...);

private:
	virtual void Write(FILE* File, int nLevel, string sText);
	virtual string Head();
	virtual string Tail();

private:
	FILE* File;
	int nLevel;
	MyMutex mtxFile;
};