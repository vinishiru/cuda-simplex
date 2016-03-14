// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>


#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // some CString constructors will be explicit

#include <atlbase.h>
#include <atlstr.h>

// TODO: reference additional headers your program requires here
#include "FileReader.h"
#include "FObjetivo.h"
#include "ILPReader.h"
#include "IReader.h"
#include "MPSReader.h"
#include "Quadro.h"
#include "Restricao.h"
#include "Stopwatch.h"
#include "StringTokenizer.h"
#include "Variavel.h"
