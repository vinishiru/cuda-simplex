#include "SimplexGPUKernels.cuh"
#include "math_functions.h"

__global__ void calcularQuocientesKernel(float *dev_vetorQuocientes, float *dev_matrizSuperior, int colunaPerm, int totalColunas, int totalLinhas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  //evitar operação em endereço invalido
  if (i > totalLinhas)
    return;

  //controle que evita divisao por 0
  if (dev_matrizSuperior[i * totalColunas + colunaPerm] == 0)
    //aplicar valor negativo para que seja desconsiderado na avaliacao do menor quociente positivo
    dev_vetorQuocientes[i] = -1;
  else
    dev_vetorQuocientes[i] = dev_matrizSuperior[i * totalColunas] / dev_matrizSuperior[i * totalColunas + colunaPerm];
}

void SimplexGPUKernels::executarCalcularQuocientesKernel(int numThreads, float *dev_vetorQuocientes, float *dev_matrizSuperior, int colunaPerm, int totalColunas, int totalLinhas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

  calcularQuocientesKernel << <blocksPerGrid, threadsPerBlock >> >(dev_vetorQuocientes, dev_matrizSuperior, colunaPerm, totalColunas, totalLinhas);
  cudaDeviceSynchronize();
}


__global__ void copiarLinhaColunaPermissiveis(float *dev_matrizSuperior, float *dev_linhaPerm, float *dev_colunaPerm, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  //os demais elementos que nao pertencem a linha ou coluna
  //calcular indices correspondentes da subcelula superior da linha
  //e coluna do elemento permissivel
  if (i < totalColunas){
    int iLinhaPerm = linhaPerm * totalColunas + (i % totalColunas);
    dev_linhaPerm[i] = dev_matrizSuperior[iLinhaPerm];
  }

  if (i < totalLinhas){
    int iColunaPerm = i * totalColunas + colunaPerm;
    dev_colunaPerm[i] = dev_matrizSuperior[iColunaPerm];
  }

}

void SimplexGPUKernels::executarCopiarLinhaColunaPermissiveis(int numThreads, float *dev_matrizSuperior, float *dev_linhaPerm, float *dev_colunaPerm, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  copiarLinhaColunaPermissiveis << <blocksPerGrid, threadsPerBlock >> >(dev_matrizSuperior, dev_linhaPerm, dev_colunaPerm, linhaPerm, colunaPerm, totalColunas, totalLinhas);
  cudaDeviceSynchronize();
}


__global__ void calculoAlgoritmoTroca(float ep, float *dev_matrizSuperior, float *dev_linhaPerm, float *dev_colunaPerm, int linhaPerm, int colunaPerm, int totalColunas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  //calcular o indice efetivo de calculo
  int iLinha = ((int)(i / totalColunas));
  int iColuna = i % totalColunas;

  //se o indice pertencer a linha ou coluna permissiveis
  if (iLinha == linhaPerm || iColuna == colunaPerm){
    //se o indice for o elemento permissivel
    if (iLinha == linhaPerm && iColuna == colunaPerm)
      dev_matrizSuperior[i] = 1 / ep;
    else if (iLinha == linhaPerm && iColuna != colunaPerm)
      //se for elemento da linha permissivel
      dev_matrizSuperior[i] = dev_matrizSuperior[i] * (1 / ep);
    else if (iLinha != linhaPerm && iColuna == colunaPerm){
      //se for elemento da coluna permissivel
      dev_matrizSuperior[i] = dev_matrizSuperior[i] * (-1 / ep);
    }
  }
  else{
    //os demais elementos que nao pertencem a linha ou coluna
    //calcular indices correspondentes da subcelula superior da linha e coluna do elemento permissivel
    /*int iLinhaSCS = linhaPerm * totalColunas + (i % totalColunas);
    int iColunaSCS = ((int)(i / totalColunas)) * totalColunas + colunaPerm;
    */
    //calcular elemento da SCI da coluna permitida
    float eleColPerm = dev_colunaPerm[iLinha] * (-1 / ep);

    //calcular elemento da SCI atual
    float eleSCI = dev_linhaPerm[iColuna] * eleColPerm;

    //somar elemento da SCI inferior virtual com o elemento da posicao atual
    dev_matrizSuperior[i] = dev_matrizSuperior[i] + eleSCI;
  }

}

void SimplexGPUKernels::executarCalculoAlgoritmoTroca(int numThreads, float ep, float *dev_matrizSuperior, float *dev_linhaPerm, float *dev_colunaPerm, int linhaPerm, int colunaPerm, int totalColunas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  calculoAlgoritmoTroca << <blocksPerGrid, threadsPerBlock >> >(ep, dev_matrizSuperior, dev_linhaPerm, dev_colunaPerm, linhaPerm, colunaPerm, totalColunas);
  cudaDeviceSynchronize();
}

