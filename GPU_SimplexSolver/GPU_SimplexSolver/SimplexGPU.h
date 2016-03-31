#ifndef __SIMPLEXGPU_H_INCLUDED__
#define __SIMPLEXGPU_H_INCLUDED__

#include "SimplexParallelFunctions.h"
#include "SimplexGPUKernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include <limits>
#include <stdio.h>

__global__ void calcularQuocientesKernel(float *dev_vetorQuocientes, float *dev_matrizSuperior, int colunaPerm, int totalColunas);

class SimplexGPU : public SimplexParallelFunctions {

public:

	SimplexGPU();

	float *calcularQuocientes(Quadro *quadro, int colunaPerm);
	void atualizarQuadro(Quadro *quadro, int colunaPerm, int linhaPerm);

  //Funcao otimizada para atualizacao do quadro (algoritmo de troca)
  void atualizarQuadroOtimizado(Quadro *quadro, int colunaPerm, int linhaPerm);

protected:

	float *dev_matrizSuperior;
	float *dev_matrizInferior;

	void alocarQuadroDevice(Quadro *quadro);
	bool cudaInitialized;
	void initCuda();

};

#endif