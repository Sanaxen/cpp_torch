#ifndef _ZIP_UTIL_H
#define _ZIP_UTIL_H
#pragma once
/*
	Copyright (c) 2019, Sanaxen
	All rights reserved.

	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.

	Decompression of zip (gz) by directory structure is not supported
*/

#pragma warning(disable : 4996)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string>

#include "config.h"
#ifdef USE_ZLIB
#include "../third_party/zlib/include/zlib.h"
#pragma comment(lib, "third_party/zlib/lib/zlibstatic.lib")
#endif

namespace cpp_torch
{

#define BUFLEN      16384
#define MAX_NAME_LEN 1024

#ifdef USE_ZLIB

	void gz_compress(FILE   *in, gzFile out)
	{
		char buf[BUFLEN];
		int len;
		int err;

		for (;;) {
			len = (int)fread(buf, 1, sizeof(buf), in);
			if (ferror(in)) {
				throw std::string("fread error");
				exit(1);
			}
			if (len == 0) break;

			if (gzwrite(out, buf, (unsigned)len) != len) throw std::string(gzerror(out, &err));
		}
		fclose(in);
		if (gzclose(out) != Z_OK) throw std::string("failed gzclose");
	}


	inline void gz_uncompress(gzFile in, FILE   *out)
	{
		char buf[BUFLEN];
		int len;
		int err;

		for (;;) {
			len = gzread(in, buf, sizeof(buf));
			if (len < 0) throw std::string(gzerror(in, &err));
			if (len == 0) break;

			if ((int)fwrite(buf, 1, (unsigned)len, out) != len) {
				throw std::string("failed fwrite");
			}
		}
		if (fclose(out)) throw std::string("failed fclose");

		if (gzclose(in) != Z_OK) throw std::string("failed gzclose");
	}

	inline void file_uncompress(char  *file, bool removeOrgfle = false)
	{
		char buf[MAX_NAME_LEN];
		char *infile, *outfile;
		FILE  *out;
		gzFile in;
		uInt len = (uInt)strlen(file);

		strcpy(buf, file);

		int SUFFIX_LEN = 0;
		char* ext = strstr(file, ".gz");
		if (ext == NULL) ext = strstr(file, ".GZ");
		if (ext == NULL) ext = strstr(file, ".zip");
		if (ext == NULL) ext = strstr(file, ".ZIP");
		if (ext == NULL) ext = strstr(file, ".tar");

		char exten[8];
		if (ext)
		{
			strcpy(exten, ext);
			SUFFIX_LEN = strlen(exten);
		}

		if (len > SUFFIX_LEN && strcmp(file + len - SUFFIX_LEN, exten) == 0) {
			infile = file;
			outfile = buf;
			outfile[len - 3] = '\0';
		}
		else {
			outfile = file;
			infile = buf;
			strcat(infile, exten);
		}
		in = gzopen(infile, "rb");
		if (in == NULL) {
			throw std::string("can't gzopen") + std::string(infile);
			exit(1);
		}
		out = fopen(outfile, "wb");
		if (out == NULL) {
			throw std::string("can't file open") + std::string(file);
			exit(1);
		}

		gz_uncompress(in, out);

		if (removeOrgfle)unlink(infile);
	}


	inline void file_compress( char  *file, std::string ext = std::string(".gz"), std::string mode = std::string("wb6 "), bool removeOrgfle = false)
	{
		char outfile[MAX_NAME_LEN];
		FILE  *in;
		gzFile out;

		strcpy(outfile, file);
		strcat(outfile, ext.c_str());

		in = fopen(file, "rb");
		if (in == NULL) {
			throw std::string("can't file open") + std::string(file);
			exit(1);
		}
		out = gzopen(outfile, mode.c_str());
		if (out == NULL) {
			throw std::string("can't gzopen") + std::string(outfile);
			exit(1);
		}
		gz_compress(in, out);

		if (removeOrgfle) unlink(file);
	}

	inline void file_uncompress(std::string& file, bool removeOrgfle = false)
	{
		file_uncompress((char*)file.c_str(), removeOrgfle);
	}
	inline void file_compress(std::string& file, std::string ext = std::string(".gz"), std::string mode = std::string("wb6 "), bool removeOrgfle = false)
	{
		file_compress((char*)file.c_str(), ext, mode, removeOrgfle);
	}
#else
inline void file_uncompress(char  *file, bool removeOrgfle = false)
{
#ifndef USE_ZLIB
	fprintf(stderr, "%s\n", "Not defined USE_ZLIB  \'file_uncompress\'"); fflush(stderr);
	throw "Not defined USE_ZLIB \'file_uncompress\'";
#endif
}


inline void file_compress(char  *file, std::string ext = std::string(".gz"), std::string mode = std::string("wb6 "), bool removeOrgfle = false)
{
#ifndef USE_ZLIB
	fprintf(stderr, "%s\n", "Not defined USE_ZLIB  \'file_compress\'"); fflush(stderr);
	throw "Not defined USE_ZLIB \'file_compress\'";
#endif
}

inline void file_uncompress(std::string& file, bool removeOrgfle = false)
{
#ifndef USE_ZLIB
	fprintf(stderr, "%s\n", "Not defined USE_ZLIB  \'file_uncompress\'"); fflush(stderr);
	throw "Not defined USE_ZLIB \'file_uncompress\'";
#endif
}
inline void file_compress(std::string& file, std::string ext = std::string(".gz"), std::string mode = std::string("wb6 "), bool removeOrgfle = false)
{
#ifndef USE_ZLIB
	fprintf(stderr, "%s\n", "Not defined USE_ZLIB  \'file_compress\'"); fflush(stderr);
	throw "Not defined USE_ZLIB \'file_compress\'";
#endif
}
#endif

}
#endif
