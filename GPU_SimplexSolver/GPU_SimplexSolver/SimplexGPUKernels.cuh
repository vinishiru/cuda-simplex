#ifndef __SIMPLEXGPU_KERNELS_H_INCLUDED__
#define __SIMPLEXGPU_KERNELS_H_INCLUDED__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

class SimplexGPUKernels{

public:

	static void executarCalcularQuocientesKernel(int numThreads, float *dev_vetorQuocientes,
		float *dev_matrizSuperior, int colunaPerm, int totalColunas, int totalLinhas);

	/*static void executarCalculoLinhaPermitida(int numThreads,
		float *dev_matrizSuperior, float *dev_matrizInferior, float ep, int linhaPerm, int totalColunas);
*/
	
	/*static void executarCalculoColunaPermitida(int numThreads,
		float *dev_matrizSuperior, float *dev_matrizInferior, float ep, int colunaPerm, int totalColunas);
*/
	static void executarCalculoLinhaColunaPermitida(int numThreads, float *dev_matrizSuperior, float *dev_matrizInferior,
		float ep, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas);

	static void executarCalculoQuadro(int numThreads,
		float *dev_matrizSuperior, float *dev_matrizInferior, int linhaPerm, int colunaPerm, int totalColunas);

	static void executarTransporLinhaColunaPermitida(int numThreads, float *dev_matrizSuperior,	float *dev_matrizInferior,
		int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas);

};
#endif