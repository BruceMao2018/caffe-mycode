#pragma once

#include "Platform.h"

class Utility
{
public:
	static string Now();
	static string Format(string format, ...);
	static bool FileIsExist(const char* filename);

	static string Trim(const string &s);
	static vector<string> Split(const string& s, const string& delim) ;

	static void toLower(char* src, int length);
	static void toUpper(char* src, int length);
	static bool isNumber(const char* src);
	static void GetFiles(vector<string>& files, string path);
	static vector<string> GetLines(string filename);
	static string toLower(string src);
	static string toUpper(string src);
	static int String2Int(const char* szValue, int nStart, int nEnd);
	static long GetPrivateProfileLong(const char* lpAppName, const char* lpKeyName, int nDefault, const char* lpFileName);
	static unsigned int GetPrivateProfileString(const char* lpAppName, const char* lpKeyName, const char* lpDefault, char* lpReturnedString, unsigned int nSize, const char* lpFileName);
};