#ifndef __SIMPLEXGPU_KERNELS_H_INCLUDED__
#define __SIMPLEXGPU_KERNELS_H_INCLUDED__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

class SimplexGPUKernels{

public:

  static void executarCalcularQuocientesKernel(int numThreads, float *dev_vetorQuocientes,
    float *dev_matrizSuperior, int colunaPerm, int totalColunas, int totalLinhas);


  static void executarCopiarLinhaColunaPermissiveis(int numThreads, float *dev_matrizSuperior, float *dev_linhaPerm,
    float *dev_colunaPerm, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas);


  static void executarCalculoAlgoritmoTroca(int numThreads, float ep, float *dev_matrizSuperior,
    float *dev_linhaPerm, float *dev_colunaPerm, int linhaPerm, int colunaPerm, int totalColunas);
};
#endif