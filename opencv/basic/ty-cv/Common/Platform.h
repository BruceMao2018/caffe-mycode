#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

#ifdef WIN32
#include <windows.h>
#include <time.h>
#include <io.h>
#include <conio.h>
#include <tchar.h>
#include <direct.h>
#include <WinSock.h>
typedef int socklen_t;
#endif

#ifdef LINUX
#include <unistd.h>
#include <arpa/inet.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <dirent.h>

#include <stdarg.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <pthread.h>
#include <dlfcn.h>
#endif

#define MAX_BUF_LEN (256)

#ifdef WIN32
#define	THREAD_TYPE DWORD WINAPI
#endif

#ifdef LINUX
#define	THREAD_TYPE void*
typedef int  BOOL;
#define FALSE (0)
#define TRUE (1)
#define	LPVOID void*
typedef THREAD_TYPE (*LPTHREAD_START_ROUTINE) (LPVOID);
typedef pthread_t HANDLE;
typedef void* HINSTANCE;

typedef int SOCKET;
typedef struct sockaddr_in SOCKADDR_IN;
typedef struct sockaddr SOCKADDR;
#define INVALID_SOCKET  (SOCKET)(~0)
#define SOCKET_ERROR            (-1)
#endif

#ifdef LINUX
void Sleep(int ms);
HINSTANCE LoadLibrary(const char* filename);
LPVOID GetProcAddress(HINSTANCE hDll, const char* funname);
BOOL FreeLibrary(HINSTANCE hDll);
unsigned long GetTickCount();
BOOL CloseHandle(HANDLE hThreadId);
#endif

BOOL CreateThread(HANDLE& hThreadId, LPTHREAD_START_ROUTINE fun, LPVOID pParam);
void WaitForThread(HANDLE hThreadId);
void TerminateThread(HANDLE hThreadId);