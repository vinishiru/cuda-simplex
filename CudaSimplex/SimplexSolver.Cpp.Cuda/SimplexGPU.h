#ifndef __SIMPLEXGPU_H_INCLUDED__
#define __SIMPLEXGPU_H_INCLUDED__

#include "SimplexGPUKernels.cuh"
#include "stdafx.h"

__global__ void calcularQuocientesKernel(float *dev_vetorQuocientes, float *dev_matrizSuperior, int colunaPerm, int totalColunas);

class SimplexGPU {

public:

  SimplexGPU();

  float *calcularQuocientes(Quadro *quadro, int colunaPerm);
  void atualizarQuadro(Quadro *quadro, int colunaPerm, int linhaPerm);

protected:

  float *dev_matriz;
  float *dev_linhaPerm;

  bool matrizAlocada;

  bool linhaPermAlocada;

  //Funcao para alocar a matriz superior no device
  //a alocacao sera feita somente na primeira vez
  //pois a partir da segunda iterecao, a matriz ja esta sincronizada
  //com a matriz do host
  void alocarQuadroDevice(Quadro *quadro);

  void copiarLinhaPermissivel(Quadro *quadro, int linhaPerm);

  bool cudaInitialized;
  void initCuda();

};

#endif