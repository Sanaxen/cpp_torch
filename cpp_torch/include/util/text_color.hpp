#ifndef _TEXT_COLOR_HPP
#define _TEXT_COLOR_HPP

#define NOMINMAX
#include <Windows.h>
#include <stdarg.h>

#define USE_WINDOWS
namespace cpp_torch
{

#ifdef USE_WINDOWS
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

		inline WORD getColorAttr(std::string& c, bool intensity = true) 
		{
			if (intensity)
			{
				if (c == "RED") return FOREGROUND_RED | FOREGROUND_INTENSITY;
				if (c == "GREEN")  return FOREGROUND_GREEN | FOREGROUND_INTENSITY;
				if (c == "BLUE")  return FOREGROUND_BLUE | FOREGROUND_INTENSITY;
				if (c == "YELLOW")  return FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY;
			}
			else
			{
				if (c == "RED") return FOREGROUND_RED;
				if (c == "GREEN")  return FOREGROUND_GREEN;
				if (c == "BLUE")  return FOREGROUND_BLUE;
				if (c == "YELLOW")  return FOREGROUND_GREEN | FOREGROUND_RED;
			}
			return 0;
		}
		inline WORD getColorAttr(char* c, bool intensity = true)
		{
			return getColorAttr(std::string(c), intensity);
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
