// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <limits>

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // some CString constructors will be explicit

#include <atlbase.h>
#include <atlstr.h>
#include <vector>

// TODO: reference additional headers your program requires here
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include "Stopwatch.h"
#include "FObjetivo.h"
#include "Quadro.h"
#include "MPSReader.h"
#include "SimplexGPU.h"
#include "SimplexGPUKernels.cuh"

#include "SimplexSolver.h"


