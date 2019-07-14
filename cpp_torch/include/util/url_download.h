#ifndef _URL_DOWNLOAD_FILE_H
#define _URL_DOWNLOAD_FILE_H
#pragma once
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "config.h"
#include<iostream>
#ifdef USE_WINDOWS
#include<windows.h>
#include <tchar.h>
#include <atlstr.h>
#include <urlmon.h>
#pragma comment(lib, "urlmon.lib")
#pragma comment(lib,"wininet.lib")
#else
#error Not supported url_download.h
#endif

namespace cpp_torch
{
	using namespace std;

	/*
	 * @param url            [in] Uniform Resource Locator (download file)
	 * @param download_file  [in] output file path
	 */
	int url_download(const char* url, const char* download_file)
	{
#ifndef USE_WINDOWS
		fprintf(stderr, "%s\n", "Not supported \'url_download\'"); fflush(stderr);
		throw "Not supported \'url_download\'";
#endif
		HRESULT hr;
		USES_CONVERSION;
		LPCTSTR Url = A2T(url), File = A2T(download_file);
		hr = URLDownloadToFile(0, Url, File, 0, 0);
		switch (hr)
		{
		case S_OK:
			cout << "Successful download:[" << download_file << "]\n";
			return 0;
			break;
		case E_OUTOFMEMORY:
			cout << "Out of memory error\n";
			return -1;
			break;
		case INET_E_DOWNLOAD_FAILURE:
			cout << "Cannot access server data:[" << url << "]\n";
			return -2;
			break;
			cout << "Unknown error\n";
			return -3;
			break;
		}
		return -99;
	}
	int url_download(std::string& url, std::string& download_file)
	{
		return url_download(url.c_str(), download_file.c_str());
	}

}

#endif
