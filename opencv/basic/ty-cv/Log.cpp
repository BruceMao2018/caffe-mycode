#include "Log.h"

Log::Log()
{
	File = NULL;
	nLevel = 0;
}

Log::~Log()
{
}

BOOL Log::Init(string strFileName, int nLevel)
{
	this->nLevel = nLevel;
	return ((File = fopen(strFileName.c_str(), "wt")) != NULL);
}

void Log::UnInit()
{
	if(File)
	{
		fclose(File);
		File = NULL;
	}
}

void Log::Write(const char* szFormat, ...)
{
	MyAutoLock autoLock((MyMutex*)&mtxFile);

	va_list ap;
	va_start(ap, szFormat);

	char szText[4096] = {0};
	vsprintf(szText, szFormat, ap);

	va_end(ap);

	Write(File, nLevel, Head() + string(szText) + Tail());
}

void Log::Write_noaffix(const char* szFormat, ...)
{
	MyAutoLock autoLock((MyMutex*)&mtxFile);

	va_list ap;
	va_start(ap, szFormat);

	char szText[4096] = { 0 };
	vsprintf(szText, szFormat, ap);

	va_end(ap);

	Write(File, nLevel, string(szText));
}

void Log::Write(FILE* File, int nLevel, string sText)
{
	if(nLevel == 3)
	{
		if(File) fputs(sText.c_str(), File);
		fputs(sText.c_str(), stdout);
	}
	else if(nLevel == 2)
	{
		if(File) fputs(sText.c_str(), File);
	}
	else if(nLevel == 1)
	{
		fputs(sText.c_str(), stdout);
	}
	else {}

	if(File) fflush(File);
}

string Log::Head()
{
	return string("");
}

string Log::Tail()
{
	return string("");
}