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
#include "util/tar_util.h"

namespace cpp_torch
{

	inline void url_download_dataSet(std::string& url, std::vector<std::string>& files, std::string& dir)
	{

		std::cout << "download...";
		for (auto file : files)
		{
			cpp_torch::url_download(url + file, dir + file);
			cpp_torch::file_uncompress(dir + file, true);

			if (strstr(file.c_str(), ".tar.gz") || strstr(file.c_str(), "TAR.GZ"))
			{
				char infile[1024];
				strcpy(infile, dir.c_str());

				strcat(infile, file.c_str());
				char* ext = strstr(infile, ".tar");
				if (ext == NULL) ext = strstr(infile, ".TAR");
				*ext = '\0';
				strcat(infile, ".tar");

				mtar_t tar;
				mtar_header_t h;

				/* Open archive for reading */
				mtar_open(&tar, infile, "r");

				std::vector<mtar_header_t> mtar_headers;
				/* Print all file names and sizes */
				while ((mtar_read_header(&tar, &h)) != MTAR_ENULLRECORD) {
					printf("%s (%d bytes)\n", h.name, h.size);
					mtar_headers.push_back(h);
					mtar_next(&tar);
				}

				std::string dir_path = "";
				for (int i = 0; i < mtar_headers.size(); i++)
				{
					char *p = NULL;
					mtar_header_t h;

					if (mtar_headers[i].size == 0 )
					{
						if (dir_path == "") dir_path = mtar_headers[i].name;
						continue;
					}
					/* Load and print contents of file "test.txt" */
					mtar_find(&tar, mtar_headers[i].name, &h);
					p = (char*)calloc(1, h.size);
					mtar_read_data(&tar, p, h.size);

					char outfile[1024];
					strcpy(outfile, mtar_headers[i].name);
					if (dir_path != "")
					{
						char* d = strstr(outfile, (char*)dir_path.c_str());
						strcpy(outfile, d+strlen(dir_path.c_str()));
					}
					char tmp[1024];
					strcpy(tmp, dir.c_str());
					strcat(tmp, outfile);
					strcpy(outfile, tmp);

					FILE* fp = fopen(outfile, "wb");
					if (fp)
					{
						unsigned res = fwrite(p, 1, h.size, fp);
						fclose(fp);
					}
					if ( p ) free(p);
				}

				/* Close archive */
				mtar_close(&tar);
				unlink(infile);
			}
		}
		std::cout << "finish" << std::endl;

	}
	inline void url_download_dataSet(char* url, std::vector<std::string>& files, char* dir)
	{
		url_download_dataSet(std::string(url), files, std::string(dir));
	}

}
#endif
