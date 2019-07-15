#ifndef _DirectryTool_h
#define _DirectryTool_h
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef USE_WINDOWS

#include "windows.h"

#include <imagehlp.h>					//MakeSureDirectoryPathExists
#pragma comment(lib, "imagehlp.lib")

#include <shlwapi.h>					//PathIsDirectoryA
#pragma comment(lib, "shlwapi.lib")


namespace cpp_torch {
class DirectryTool
{
public:
	bool ExistDir(char* dirname)
	{
		return (bool)PathIsDirectoryA(dirname);

	}

	DirectryTool() {}
	DirectryTool(char* dirname) { MakeDir(dirname); }

	bool MakeDir(char* dirname)
	{
		char tmp[256];
		strcpy(tmp, dirname);
		char* p = tmp;
		while (*p)
		{
			if (*p == '/') *p = '\\';
			p++;
		}
		//if ( ExistDir(dirname) )
		//{
		//	return true;
		//}
		//if (MakeSureDirectoryPathExists(".\\aaa\\bbb\\")) {
		//	puts("succeeded.");
		//}
		//else {
		//	puts("failed");
		//}	
		printf("MakeDir[%s]\n", tmp);
		return (bool)MakeSureDirectoryPathExists(tmp);
	}
};
}
#endif

#endif