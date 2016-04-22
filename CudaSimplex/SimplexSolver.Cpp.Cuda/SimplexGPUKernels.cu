#include "stdafx.h"

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
    dev_vetorQuocientes[i] = dev_matrizSuperior[i * totalColunas + totalColunas - 1] / dev_matrizSuperior[i * totalColunas + colunaPerm];
}

void SimplexGPUKernels::executarCalcularQuocientesKernel(int numThreads, float *dev_vetorQuocientes, float *dev_matrizSuperior, int colunaPerm, int totalColunas, int totalLinhas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 1024;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

  calcularQuocientesKernel << <blocksPerGrid, threadsPerBlock >> >(dev_vetorQuocientes, dev_matrizSuperior, colunaPerm, totalColunas, totalLinhas);
  cudaDeviceSynchronize();
}


__global__ void calculoAlgoritmoTroca(float *dev_matrizSuperior, int linhaPerm, int colunaPerm, int totalColunas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  float fatorAnulador = 0.0;

  //se for indice da linha permissivel, desconsiderar
  if (i == linhaPerm)
    return;

  //computar fator anulador da respectiva linha
  fatorAnulador = dev_matrizSuperior[i * totalColunas + colunaPerm] * (-1);

  //calcular os valores dos elementos da linha usando o fator anulador coletado
  for (int coluna = 0; coluna < totalColunas; coluna++){
    //o valor da coluna permissivel sera 0
    if (coluna == colunaPerm)
      dev_matrizSuperior[i * totalColunas + coluna] = 0;
    else
      //os demais valores devem respeitar a equacao
      //Valor = FatorAnulador * ValorLinhaPerm + LinhaAtual;
      dev_matrizSuperior[i * totalColunas + coluna] = fatorAnulador *
      dev_matrizSuperior[linhaPerm *totalColunas + coluna] +
      dev_matrizSuperior[i * totalColunas + coluna];
  }

}

void SimplexGPUKernels::executarCalculoAlgoritmoTroca(int numThreads, float *dev_matrizSuperior, int linhaPerm, int colunaPerm, int totalColunas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 1024;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  calculoAlgoritmoTroca << <blocksPerGrid, threadsPerBlock >> >(dev_matrizSuperior, linhaPerm, colunaPerm, totalColunas);
  cudaDeviceSynchronize();
}

