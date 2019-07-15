#ifndef _UTF8_PRINTF_HPP

#define _UTF8_PRINTF_HPP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_WINDOWS
#include <Windows.h>

namespace cpp_torch {
class utf8str
{
	unsigned char *bufUTF8;
public:

	utf8str()
	{
		bufUTF8 = 0;
	}
	~utf8str()
	{
		if ( bufUTF8 ) delete [] bufUTF8;
		bufUTF8 = 0;
	}

	/*
	WindowsAPIのMultiByteToWideChar()関数を利用して、
	ShiftJisコードをUnicodeへ変換。
	ShiftJisコードからUTF8に変換するには、Unicode経由しなければなりません。 
	*/
	//inline void ShiftJisToUTF8(char* szShiftJis)
	//{
	//	if ( bufUTF8 ) delete [] bufUTF8;
	//	bufUTF8 = 0;

	//	int len = strlen(szShiftJis)+1;

	//	wchar_t *bufUnicode = new wchar_t[8*len];
	//	memset(bufUnicode, '\0', sizeof(wchar_t)*8*len);
 //   
	//	// まずUniocdeに変換する
	//	// サイズを計算する
	//	int iLenUnicode = MultiByteToWideChar(CP_ACP, 0, szShiftJis, strlen(szShiftJis)+1,  NULL, 0);
	//	if (iLenUnicode <= 8*len)
	//	{
	//		MultiByteToWideChar(CP_ACP, 0, szShiftJis, strlen(szShiftJis)+1, bufUnicode, MAX_PATH);
	//		// 次に、UniocdeからUTF8に変換する
	//		// サイズを計算する
	//		int iLenUtf8 = WideCharToMultiByte(CP_UTF8, 0, bufUnicode, iLenUnicode, NULL, 0, NULL, NULL);

	//		bufUTF8 = new char[iLenUtf8+1];
	//		memset(bufUTF8, '\0', sizeof(char)*(iLenUtf8+1));
	//		WideCharToMultiByte(CP_UTF8, 0, bufUnicode, iLenUnicode, bufUTF8, sizeof(bufUTF8),  NULL, NULL);
	//	}
	//	delete [] bufUnicode;
	//}


	BOOL convSJIStoUTF8( char* pSource, unsigned char* pDist, int* pSize )
	{	
		*pSize = 0; 	
		// Convert SJIS -> UTF-16	
		const int nSize = ::MultiByteToWideChar( CP_ACP, 0, (LPCSTR)pSource, -1, NULL, 0 );
		unsigned char* buffUtf16 = new unsigned char[ nSize * 2 + 2 ];	
		::MultiByteToWideChar( CP_ACP, 0, (LPCSTR)pSource, -1, (LPWSTR)buffUtf16, nSize ); 	
		// Convert UTF-16 -> UTF-8	
		const int nSizeUtf8 = ::WideCharToMultiByte( CP_UTF8, 0, (LPCWSTR)buffUtf16, -1, NULL, 0, NULL, NULL );	
		if ( !pDist ) 
		{		
			*pSize = nSizeUtf8;		
			delete [] buffUtf16;		
			return TRUE;	
		}

		unsigned char* buffUtf8 = new unsigned char[ nSizeUtf8 * 2 ];	
		ZeroMemory( buffUtf8, nSizeUtf8 * 2 );	
		::WideCharToMultiByte( CP_UTF8, 0, (LPCWSTR)buffUtf16, -1, (LPSTR)buffUtf8, nSizeUtf8, NULL, NULL ); 	
		*pSize = strlen( (char*)buffUtf8 );	
		memcpy( pDist, buffUtf8, *pSize ); 	
		
		delete [] buffUtf16;	
		delete [] buffUtf8; 	
		return TRUE;
	} 
	
	/* * convert: sjis -> utf8 */
	BOOL sjis2utf8(char* source) 
	{	
		// Calculate result size	
		int size = 0;	
		convSJIStoUTF8( source, NULL, &size ); 	
		// Peform convert	
		bufUTF8 = new unsigned char[ size + 1 ];	
		ZeroMemory( bufUTF8, size + 1 );	
		convSJIStoUTF8( source, bufUTF8, &size ); 	
		return TRUE;
	}

	inline void printf(char* format, ...)
	{
		va_list	argp;
		char pszBuf[ 4096];
		va_start(argp, format);
		vsnprintf( pszBuf, 4096, format, argp);
		va_end(argp);

		sjis2utf8(pszBuf);
		::printf("%s", bufUTF8);
		if ( bufUTF8 ) delete [] bufUTF8;
		bufUTF8 = 0;

	}
	inline void fprintf(FILE* fp, char* format, ...)
	{
		va_list	argp;
		char pszBuf[ 4096];
		va_start(argp, format);
		vsnprintf( pszBuf, 4096, format, argp);
		va_end(argp);
		//::fprintf(fp, "%s", pszBuf); fflush(fp);
		//return;

		sjis2utf8(pszBuf);
		if ( fp != NULL )
		{
			::fprintf(fp, "%s", bufUTF8); fflush(fp);
		}
		if ( bufUTF8 ) delete [] bufUTF8;
		bufUTF8 = 0;
	}
};


inline std::string WStringToString
(
    std::wstring oWString
)
{
    // wstring → SJIS
    int iBufferSize = WideCharToMultiByte( CP_OEMCP, 0, oWString.c_str()
        , -1, (char *)NULL, 0, NULL, NULL );
 
    // バッファの取得
    CHAR* cpMultiByte = new CHAR[ iBufferSize ];
 
    // wstring → SJIS
    WideCharToMultiByte( CP_OEMCP, 0, oWString.c_str(), -1, cpMultiByte
        , iBufferSize, NULL, NULL );
 
    // stringの生成
    std::string oRet( cpMultiByte, cpMultiByte + iBufferSize - 1 );
 
    // バッファの破棄
    delete [] cpMultiByte;
 
    // 変換結果を返す
    return( oRet );
}
}
#endif
#endif