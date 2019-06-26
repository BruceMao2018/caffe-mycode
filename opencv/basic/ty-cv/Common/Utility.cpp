#include "Utility.h"

string Utility::Now()
{
	time_t t = time(NULL); 
	struct tm* local = localtime(&t); 

	local->tm_year = local->tm_year + 1900;
	local->tm_mon ++;

	return Format("%.4d-%.2d-%.2d %.2d:%.2d:%.2d", local->tm_year, local->tm_mon, local->tm_mday, local->tm_hour, local->tm_min, local->tm_sec);
}

string Utility::Format(string format, ...)
{	
	va_list ap;
	va_start(ap, format);

	char text[1024] = {0};
	vsprintf(text, format.c_str(), ap);

	va_end(ap);

	return string(text);
}

bool Utility::FileIsExist(const char* filename)
{
	if (filename == NULL) return false;
	else
	{
		FILE* fp = fopen(filename, "r");

		if (fp == NULL)
			return false;
		else
		{
			fclose(fp);
			return true;
		}
	}
}

string Utility::Trim(const string &s)   
{
	char dst[MAX_BUF_LEN] = {0};
	strcpy(dst, s.c_str());

	while(dst[0] == ' ') 
	{
		for(int i=0; i<=strlen(dst); i++)
			dst[i] = dst[i+1];
	}

	while(dst[strlen(dst)-1] == '\r' || dst[strlen(dst)-1] == '\n' || dst[strlen(dst)-1] == ' ') 
	{
		dst[strlen(dst)-1] = 0;
	}
	
	return string(dst);
}  

vector<string> Utility::Split(const string& s, const string& delim) 
{
	vector<string> dst;
	size_t last = 0;  
	size_t X = s.find_first_of(delim, last);

	while(X != string::npos)  
	{
		dst.push_back(Trim(s.substr(last, X-last)));  
		last = X+delim.size();  
		X = s.find_first_of(delim, last);
	}
	
	if(X-last > 0)
	{
		dst.push_back(Trim(s.substr(last,X-last)));  
	}
	
	return dst;
}


void Utility::toLower(char* src, int length)
{
	for(int X=0; X<length; X++)
		src[X] = tolower(src[X]);
}

void Utility::toUpper(char* src, int length)
{
	for(int X=0; X<length; X++)
		src[X] = toupper(src[X]);
}

bool Utility::isNumber(const char* src)
{
	for(int X=0; X<strlen(src); X++)
		if(src[X]>'9' || src[X]<'0') return false;
	return true;
}

void Utility::GetFiles(vector<string>& files, string path)
{
#ifdef WIN32
	intptr_t   hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst((path + "/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					Utility::GetFiles(files, path + "/" + fileinfo.name);
			}
			else files.push_back(path + "/" + fileinfo.name);
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
#endif
}

vector<string> Utility::GetLines(string filename)
{
	vector<string> lines;
	char line[MAX_BUF_LEN] = {0};

	FILE* fp = fopen(filename.c_str(), "rt");
	if (fp)
	{
		while (fgets(line, MAX_BUF_LEN, fp))
			lines.push_back(Trim(line));

		fclose(fp);
	}

	return lines;
}

string Utility::toLower(string src)
{
	char dst[MAX_BUF_LEN] = {0};

	strcpy(dst, src.c_str());
	toLower(dst, src.length());

	return dst;
}

string Utility::toUpper(string src)
{
	char dst[MAX_BUF_LEN] = {0};

	strcpy(dst, src.c_str());
	toUpper(dst, src.length());

	return dst;
}


int Utility::String2Int(const char* szValue, int nStart, int nEnd)
{
	int number = 0;
	for (int X = nStart; X <= nEnd; X++)
		number = number * 10 + (szValue[X] - '0');
	return number;
}

long Utility::GetPrivateProfileLong(const char* lpAppName, const char* lpKeyName, int nDefault, const char* lpFileName)
{
	char szTmp[MAX_BUF_LEN + 1] = { 0 };

	FILE* fp = fopen(lpFileName, "rt");
	if (fp == NULL)
		return(nDefault);

	int nfind = 0;
	while (fgets(szTmp, MAX_BUF_LEN, fp))
	{
		if (string(szTmp).find(lpAppName) != string::npos)
		{
			nfind = 1;
			break;
		}
	}
	if (nfind == 0)
	{
		fclose(fp);
		return(nDefault);
	}

	while (fgets(szTmp, MAX_BUF_LEN, fp))
	{
		vector<string> result = Split(string(szTmp), "=");
		if (result.size() == 2)
		{
			if ((strcmp(result[0].c_str(), lpKeyName) == 0) && result[1].length() >= 1)
			{
				fclose(fp);
				return atol(result[1].c_str());
			}
		}
		else
		{
			if ((string(szTmp).find("[") != string::npos) || (string(szTmp).find("]") != string::npos))
				break;
		}
	}

	fclose(fp);
	return(nDefault);
}

unsigned int Utility::GetPrivateProfileString(const char* lpAppName, const char* lpKeyName, const char* lpDefault, char* lpReturnedString, unsigned int nSize, const char* lpFileName)
{
	char szTmp[MAX_BUF_LEN + 1] = { 0 };

	FILE* fp = fopen(lpFileName, "rt");
	if (fp == NULL)
		return(0);

	int nfind = 0;
	while (fgets(szTmp, MAX_BUF_LEN, fp))
	{
		if (string(szTmp).find(lpAppName) != string::npos)
		{
			nfind = 1;
			break;
		}
	}
	if (nfind == 0)
	{
		fclose(fp);
		return(0);
	}

	while (fgets(szTmp, MAX_BUF_LEN, fp))
	{
		vector<string> result = Split(string(szTmp), "=");
		if (result.size() == 2)
		{
			if ((strcmp(result[0].c_str(), lpKeyName) == 0) && result[1].length() >= 1)
			{
				fclose(fp);
				strcpy(lpReturnedString, result[1].c_str());
				return 1;
			}
		}
		else
		{
			if ((string(szTmp).find("[") != string::npos) || (string(szTmp).find("]") != string::npos))
				break;
		}
	}

	if (lpDefault)
		strcpy(lpReturnedString, lpDefault);
	return (0);
}
