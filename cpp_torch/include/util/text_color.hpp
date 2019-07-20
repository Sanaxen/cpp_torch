/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/
#ifndef _TEXT_COLOR_HPP
#define _TEXT_COLOR_HPP

#define NOMINMAX
#include <Windows.h>
#include <stdarg.h>

#include "config.h"
namespace cpp_torch
{

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
	int console_create__ = 0;
	inline void console_create()
	{
		if (console_create__) return;

		int hConHandle;
		long lStdHandle;
		//HANDLE lStdHandle;
		CONSOLE_SCREEN_BUFFER_INFO coninfo;
		FILE *fp;
		FreeConsole(); // be sure to release possible already allocated console
		if (!AllocConsole())
		{
			fprintf(stderr, "Cannot allocate windows console!");
			return;
		}
		console_create__ = 1;
	}

#define TEXT_COLOR_STRING_MAX 512
	/*
	#define FOREGROUND_BLUE      0x0001 // text color contains blue.
	#define FOREGROUND_GREEN     0x0002 // text color contains green.
	#define FOREGROUND_RED       0x0004 // text color contains red.
	*/
	class textColor
	{
		HANDLE hStdout;
		CONSOLE_SCREEN_BUFFER_INFO Info;
		char* pszBuf;
	public:
		HANDLE getHANDLE() const
		{
			return hStdout;
		}

		inline void init()
		{
			pszBuf = new char[TEXT_COLOR_STRING_MAX];
			hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
			GetConsoleScreenBufferInfo(hStdout, &Info);

		}

		inline textColor()
		{
			init();
			color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		}
		inline textColor(WORD c)
		{
			init();
			color(c);
		}
		inline textColor(std::string& c, bool intensity = true)
		{
			init();
			color(getColorAttr(c, intensity));
		}
		inline textColor(char* c, bool intensity = true)
		{
			init();
			color(getColorAttr(c, intensity));
		}


		~textColor()
		{
			SetConsoleTextAttribute(hStdout, Info.wAttributes);
			if (pszBuf) delete[] pszBuf;
			pszBuf = 0;
		}

		inline WORD getColorAttr(std::string& c, bool front = true, bool intensity = false) 
		{
			WORD color = 0;
			if (front)
			{
				if (c == "RED")     color |= FOREGROUND_RED;
				if (c == "GREEN")   color |= FOREGROUND_GREEN;
				if (c == "BLUE")    color |= FOREGROUND_BLUE;
				if (c == "YELLOW")  color |= FOREGROUND_GREEN | FOREGROUND_RED;
				if (c == "CYAN")    color |= FOREGROUND_GREEN | FOREGROUND_BLUE;
				if (c == "MAGENTA") color |= FOREGROUND_RED | FOREGROUND_BLUE;
				if (c == "GRAY")    color |= FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;

				if (intensity)
				{
					color |= FOREGROUND_INTENSITY;
				}

				if (c == "WHITE")         color = FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
				if (c == "DARKYELLOW")    color = FOREGROUND_RED | FOREGROUND_GREEN;
				if (c == "DARKBLUE")      color = FOREGROUND_BLUE;
				if (c == "DARKGREEN")     color = FOREGROUND_GREEN;
				if (c == "DARKRED")       color = FOREGROUND_RED;
				if (c == "DARKCYAN")      color = FOREGROUND_GREEN | FOREGROUND_BLUE;
				if (c == "DARKMAGENTA")   color = FOREGROUND_RED | FOREGROUND_BLUE;
				if (c == "DARKYELLOW")    color = FOREGROUND_RED | FOREGROUND_GREEN;
			}
			else
			{
				if (c == "RED")     color |= BACKGROUND_RED;
				if (c == "GREEN")   color |= BACKGROUND_GREEN;
				if (c == "BLUE")    color |= BACKGROUND_BLUE;
				if (c == "YELLOW")  color |= BACKGROUND_GREEN | BACKGROUND_RED;
				if (c == "CYAN")    color |= BACKGROUND_GREEN | BACKGROUND_BLUE;
				if (c == "MAGENTA") color |= BACKGROUND_RED | BACKGROUND_BLUE;
				if (c == "GRAY")    color |= BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE;

				if (intensity)
				{
					color |= BACKGROUND_INTENSITY;
				}
				if (c == "WHITE")         color = BACKGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE;
				if (c == "DARKYELLOW")    color = BACKGROUND_RED | BACKGROUND_GREEN;
				if (c == "DARKBLUE")      color = BACKGROUND_BLUE;
				if (c == "DARKGREEN")     color = BACKGROUND_GREEN;
				if (c == "DARKRED")       color = BACKGROUND_RED;
				if (c == "DARKCYAN")      color = BACKGROUND_GREEN | BACKGROUND_BLUE;
				if (c == "DARKMAGENTA")   color = BACKGROUND_RED | BACKGROUND_BLUE;
				if (c == "DARKYELLOW")    color = BACKGROUND_RED | BACKGROUND_GREEN;
			}
			return color;
		}
		inline WORD getColorAttr(char* c, bool front = true, bool intensity = true)
		{
			return getColorAttr(std::string(c), front, intensity);
		}

		inline void color(WORD atter)
		{
			SetConsoleTextAttribute(hStdout, atter);
		}
		inline void reset()
		{
			SetConsoleTextAttribute(hStdout, Info.wAttributes);
			FlushFileBuffers(hStdout);
		}
		inline void printf(char* format, ...)
		{
			va_list	argp;
			va_start(argp, format);
			vsnprintf(pszBuf, TEXT_COLOR_STRING_MAX, format, argp);
			va_end(argp);
			DWORD length = 0;
			WriteConsoleA(hStdout, pszBuf, strlen(pszBuf), &length, 0);
			FlushFileBuffers(hStdout);
		}

		inline void clear_line(int linenum)
		{
			CONSOLE_SCREEN_BUFFER_INFO info;
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&info);

			color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
			for (int i = 0; i < 80 * linenum; ++i)
			{
				this->printf(" ");
			}
			this->printf("\n");
			SetConsoleCursorPosition(hStdout, info.dwCursorPosition);
		}


		CONSOLE_SCREEN_BUFFER_INFO info_tmp;
		inline void begin(char* format, ...)
		{
			va_list	argp;
			va_start(argp, format);
			vsnprintf(pszBuf, TEXT_COLOR_STRING_MAX, format, argp);
			va_end(argp);

			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&info_tmp);

			//::printf("%s", pszBuf);
			DWORD length = 0;
			WriteConsoleA(hStdout, pszBuf, strlen(pszBuf), &length, 0);
			FlushFileBuffers(hStdout);
		}
		inline void end(char* format, ...)
		{
			va_list	argp;
			va_start(argp, format);
			vsnprintf(pszBuf, TEXT_COLOR_STRING_MAX, format, argp);
			va_end(argp);

			SetConsoleCursorPosition(hStdout, info_tmp.dwCursorPosition);
			//::printf("%s", pszBuf);
			DWORD length = 0;
			WriteConsoleA(hStdout, pszBuf, strlen(pszBuf), &length, 0);
			FlushFileBuffers(hStdout);
		}
	};
#else
int console_create__ = 0;
inline void console_create()
{
	if (console_create__) return;
	console_create__ = 1;
	}

#define FOREGROUND_BLUE      0x0001 // text color contains blue.
#define FOREGROUND_GREEN     0x0002 // text color contains green.
#define FOREGROUND_RED       0x0004 // text color contains red.

class textColor
{
	FILE* hStdout;
	CONSOLE_SCREEN_BUFFER_INFO Info;
	char* pszBuf;
public:
	FILE* getHANDLE() const
	{
		return hStdout;
	}

	inline void init()
	{
		pszBuf = new char[512];
		hStdout = stdout;

	}

	inline textColor()
	{
		init();
		color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
	}
	inline textColor(WORD c)
	{
		init();
		color(c);
	}
	inline textColor(std::string& c, bool intensity = true)
	{
		init();
	}
	inline textColor(char* c, bool intensity = true)
	{
		init();
	}

	~textColor()
	{
		if (pszBuf) delete[] pszBuf;
		pszBuf = 0;
	}

	inline void color(WORD atter)
	{
	}
	inline void reset()
	{
	}
	inline void printf(char* format, ...)
	{
		va_list	argp;
		va_start(argp, format);
		vsnprintf(pszBuf, 512, format, argp);
		va_end(argp);
		int length = 0;
		fprintf(hStdout, "%s", pszBuf);
		fflush(hStdout);
	}

	inline void clear_line(int linenum)
	{
		color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		for (int i = 0; i < 80 * linenum; ++i)
		{
			this->printf(" ");
		}
		this->printf("\n");
	}


	inline void begin(char* format, ...)
	{
		va_list	argp;
		va_start(argp, format);
		vsnprintf(pszBuf, 512, format, argp);
		va_end(argp);


		::fprintf(hStdout, "%s", pszBuf);
		fflush(hStdout);
	}
	inline void end(char* format, ...)
	{
		va_list	argp;
		va_start(argp, format);
		vsnprintf(pszBuf, 512, format, argp);
		va_end(argp);

		::fprintf(hStdout, "%s", pszBuf);
		fflush(hStdout);
	}
};
#endif

}
#endif
