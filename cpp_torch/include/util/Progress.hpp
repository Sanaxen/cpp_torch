/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/
#ifndef _PROGRESS_H
#define _PROGRESS_H

#include "config.h"

#include "text_color.hpp"
namespace cpp_torch
{
	class progress_display2 {
	public:
		explicit progress_display2(size_t expected_count_,
			std::ostream &os = std::cout,
			const std::string &s1 = "\n",  // leading strings
			const std::string &s2 = "",
			const std::string &s3 = "")
			// os is hint; implementation may ignore, particularly in embedded systems
			: m_os(os), m_s1(s1), m_s2(s2), m_s3(s3) {
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			CONSOLE_SCREEN_BUFFER_INFO now;
			GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), (PCONSOLE_SCREEN_BUFFER_INFO)&now);
			display_size = now.dwSize;
#endif
			restart(expected_count_);
		}

		void end()
		{
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			if (hStdout)
			{
				fflush(stdout);
				std::cout << std::flush;
				GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&current);
				SetConsoleCursorPosition(hStdout, progress_str.dwCursorPosition);
				printf("\r");
				fflush(stdout);
				console.color(console.getColorAttr("WHITE") | console.getColorAttr("WHITE", false));
				console.printf("##################################################\n");
				console.reset();
				SetConsoleCursorPosition(hStdout, current.dwCursorPosition);
			}
#endif
		}

		void restart(size_t expected_count_, const std::string& h = "") {
			//  Effects: display appropriate scale
			//  Postconditions: count()==0, expected_count()==expected_count_
			_count = _next_tic_count = _tic = 0;
			_expected_count = expected_count_;

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			if (hStdout)
			{
				fflush(stdout);
				std::cout << std::flush;
				GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&current);
				SetConsoleCursorPosition(hStdout, progress_str.dwCursorPosition);
				printf("\r");
				fflush(stdout);
				console.color(console.getColorAttr("WHITE") | console.getColorAttr("WHITE", false));
				console.printf("##################################################\n");
				console.reset();
				SetConsoleCursorPosition(hStdout, current.dwCursorPosition);
			}
			else
			{
				hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
			}

			CONSOLE_SCREEN_BUFFER_INFO now;
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&now);
			if (now.dwCursorPosition.Y > display_size.Y - 4)
			{
				if (now.dwSize.Y >= SHRT_MAX / 100)
				{
					system("cls");
				}
				else
				{
					now.dwSize.Y += 100;
					display_size = now.dwSize;
					SetConsoleScreenBufferSize(hStdout, now.dwSize);
				}
			}
#else
			printf("\r");
			fflush(stdout);
			printf("##################################################\n");
#endif

			m_os << m_s1
				<< m_s2 << "|----|----|----|----|----|----|----|----|----|----|";
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			fflush(stdout);
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&progress_str);
#endif
			m_os << h;
			m_os << std::flush;
			printf("\r");
			if (!_expected_count) _expected_count = 1;  // prevent divide by zero
		}                                             // restart

		size_t operator+=(size_t increment) {
			//  Effects: Display appropriate progress tic if needed.
			//  Postconditions: count()== original count() + increment
			//  Returns: count().
			if ((_count += increment) >= _next_tic_count) {
				display_tic();
			}
			return _count;
		}

		size_t operator++() { return operator+=(1); }
		size_t count() const { return _count; }
		size_t expected_count() const { return _expected_count; }

	private:
		std::ostream &m_os;      // may not be present in all imps
		const std::string m_s1;  // string is more general, safer than
		const std::string m_s2;  //  const char *, and efficiency or size are
		const std::string m_s3;  //  not issues

		size_t _count, _expected_count, _next_tic_count;
		size_t _tic;
		void display_tic() {
			// use of floating point ensures that both large and small counts
			// work correctly.  static_cast<>() is also used several places
			// to suppress spurious compiler warnings.
			size_t tics_needed = static_cast<size_t>(
				(static_cast<double>(_count) / _expected_count) * 50.0);

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			console.color(console.getColorAttr("YELLOW") | console.getColorAttr("YELLOW", false));
#endif
			do {
				m_os << '*' << std::flush;
			} while (++_tic < tics_needed);
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			console.reset();
#endif

			_next_tic_count = static_cast<size_t>((_tic / 50.0) * _expected_count);
			if (_count == _expected_count) {
				if (_tic < 51) m_os << '*';
				m_os << std::endl;
			}
		}  // display_tic

		progress_display2 &operator=(const progress_display2 &) = delete;

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
		HANDLE hStdout = NULL;
		COORD display_size;
		CONSOLE_SCREEN_BUFFER_INFO progress_str;
		CONSOLE_SCREEN_BUFFER_INFO progress_cur;
		CONSOLE_SCREEN_BUFFER_INFO current;
		textColor console;
#endif

	};

	class progress_display {
	public:
		explicit progress_display(size_t expected_count_,
			std::ostream &os = std::cout,
			const std::string &s1 = "\n",  // leading strings
			const std::string &s2 = "",
			const std::string &s3 = "")
			// os is hint; implementation may ignore, particularly in embedded systems
			: m_os(os), m_s1(s1), m_s2(s2), m_s3(s3) {
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			CONSOLE_SCREEN_BUFFER_INFO now;
			GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), (PCONSOLE_SCREEN_BUFFER_INFO)&now);
			display_size = now.dwSize;
#endif
			restart(expected_count_);
		}

		void restart(size_t expected_count_, const std::string& h = "") {
			//  Effects: display appropriate scale
			//  Postconditions: count()==0, expected_count()==expected_count_
			_count = _next_tic_count = _tic = 0;
			_expected_count = expected_count_;

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			if (hStdout)
			{
				fflush(stdout);
				std::cout << std::flush;
				GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&current);
				SetConsoleCursorPosition(hStdout, progress_str.dwCursorPosition);
				printf("\r");
				fflush(stdout);
				console.color(FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_GREEN | BACKGROUND_INTENSITY);
				console.printf("##################################################", h.c_str());
				console.color(FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
				console.reset();
				console.printf(" %s\n", h.c_str());
				SetConsoleCursorPosition(hStdout, current.dwCursorPosition);
			}
			else
			{
				hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
			}

			CONSOLE_SCREEN_BUFFER_INFO now;
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&now);
			if (now.dwCursorPosition.Y > display_size.Y - 4)
			{
				if (now.dwSize.Y >= SHRT_MAX/100)
				{
					system("cls");
				}
				else
				{
					now.dwSize.Y += 100;
					display_size = now.dwSize;
					SetConsoleScreenBufferSize(hStdout, now.dwSize);
				}
			}
#else
			printf("\r");
			printf("################################################## %s\n", h.c_str());
#endif

			m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
				<< m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
				<< std::endl  // endl implies flush, which ensures display
				<< m_s3;
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			fflush(stdout);
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&progress_str);
#endif
			if (!_expected_count) _expected_count = 1;  // prevent divide by zero
		}                                             // restart

		size_t operator+=(size_t increment) {
			//  Effects: Display appropriate progress tic if needed.
			//  Postconditions: count()== original count() + increment
			//  Returns: count().
			if ((_count += increment) >= _next_tic_count) {
				display_tic();
			}
			return _count;
		}

		size_t operator++() { return operator+=(1); }
		size_t count() const { return _count; }
		size_t expected_count() const { return _expected_count; }

	private:
		std::ostream &m_os;      // may not be present in all imps
		const std::string m_s1;  // string is more general, safer than
		const std::string m_s2;  //  const char *, and efficiency or size are
		const std::string m_s3;  //  not issues

		size_t _count, _expected_count, _next_tic_count;
		size_t _tic;
		void display_tic() {
			// use of floating point ensures that both large and small counts
			// work correctly.  static_cast<>() is also used several places
			// to suppress spurious compiler warnings.
			size_t tics_needed = static_cast<size_t>(
				(static_cast<double>(_count) / _expected_count) * 50.0);

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			console.color(FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_INTENSITY);
			//console.color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN /*| BACKGROUND_INTENSITY*/);
#endif
			do {
				m_os << '*' << std::flush;
			} while (++_tic < tics_needed);
#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
			console.reset();
#endif

			_next_tic_count = static_cast<size_t>((_tic / 50.0) * _expected_count);
			if (_count == _expected_count) {
				if (_tic < 51) m_os << '*';
				m_os << std::endl;
			}
		}  // display_tic

		progress_display &operator=(const progress_display &) = delete;

#if  defined(USE_WINDOWS) && defined(USE_COLOR_CONSOLE)
		HANDLE hStdout = NULL;
		COORD display_size;
		CONSOLE_SCREEN_BUFFER_INFO progress_str;
		CONSOLE_SCREEN_BUFFER_INFO progress_cur;
		CONSOLE_SCREEN_BUFFER_INFO current;
		textColor console;
#endif

	};

#if 0
	class ProgressPrint
	{
		HANDLE hStdout;
		CONSOLE_SCREEN_BUFFER_INFO stt;
		CONSOLE_SCREEN_BUFFER_INFO cur;
		textColor console;

		int counter_max;
		int counter;
		int print_num;
		std::chrono::high_resolution_clock::time_point start_time;
	public:

		inline void init()
		{
			counter = 0;
			print_num = 0;
			hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		}
		ProgressPrint(int n)
		{
			init();
			counter_max = n;
		}
		ProgressPrint()
		{
			init();
		}

		inline void newbar()
		{
			start_time = std::chrono::high_resolution_clock::now();
			print_num = 0;
			counter = 0;
			console.color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
			printf("0%%   10   20   30   40   50   60   70   80   90   100%%\n");
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&stt);
			GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&cur);
			printf("|----|----|----|----|----|----|----|----|----|----|\n\r"); fflush(stdout);
			console.reset();
		}
		inline void start()
		{
			start_time = std::chrono::high_resolution_clock::now();
			newbar();
		}
		inline void end()
		{
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			float sec = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now()- start_time).count();

			printf("\r");
			SetConsoleCursorPosition(hStdout, stt.dwCursorPosition);
			console.color(FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_GREEN | BACKGROUND_INTENSITY);
			console.printf("###################################################");
			console.color(FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
			//console.clear_line(5);
			//console.printf("\n %.3f msec\n", sec);
			console.printf("\n\n");
			console.reset();
			fflush(stdout);
		}

		inline void print()
		{
			if (print_num == 51) return;
			counter++;

			int c = 51.0*(float)counter / (float)counter_max;
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

			if (c > print_num)
			{
				print_num++;
				SetConsoleCursorPosition(hStdout, cur.dwCursorPosition);
				console.color(FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED| FOREGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_INTENSITY);
				//console.color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_GREEN /*| BACKGROUND_INTENSITY*/);
				console.printf(" ");
				GetConsoleScreenBufferInfo(hStdout, (PCONSOLE_SCREEN_BUFFER_INFO)&cur);
				console.reset();

				console.color(FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
				console.reset();
				fflush(stdout);
				console.reset();
			}
		}
	};

	class measurement_time
	{
		std::chrono::system_clock::time_point start_;
		std::chrono::system_clock::time_point end;
	public:
		measurement_time()
		{
			start_ = std::chrono::system_clock::now();
		}

		inline void start()
		{
			start_ = std::chrono::system_clock::now();
		}
		inline void stop()
		{
			end = std::chrono::system_clock::now();  // �v���I������

			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
			printf("%f[milliseconds]\n", elapsed);
		}

	};

#endif

}
#endif