#pragma once

#include "stdafx.h"

class SimplexGPUKernels{

public:

  static void executarCalcularQuocientesKernel(int numThreads, float *dev_vetorQuocientes,
    float *dev_matrizSuperior, int colunaPerm, int totalColunas, int totalLinhas);

  static void executarCalculoAlgoritmoTroca(int numThreads, float *dev_matrizSuperior,
    int linhaPerm, int colunaPerm, int totalColunas);
};