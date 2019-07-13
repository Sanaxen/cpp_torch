#ifndef _DOWNLOAD_DATA_SET_H
#define _DOWNLOAD_DATA_SET_H
#pragma once
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include <vector>
#include <string>

#include "util/url_download.h"
#include "util/zip_util.h"

inline void url_download_dataSet(std::string& url, std::vector<std::string>& files, std::string& dir)
{

	std::cout << "download...";
	for (auto file : files)
	{
		cpp_torch::url_download(url + file, dir + file);
		cpp_torch::file_uncompress(dir + file, true);
	}
	std::cout << "finish" << std::endl;

}
inline void url_download_dataSet(char* url, std::vector<std::string>& files, char* dir)
{
	url_download_dataSet(std::string(url), files, std::string(dir));
}


#endif
