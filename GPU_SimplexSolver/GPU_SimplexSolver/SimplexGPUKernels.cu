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

//Metodo antigo para calculo das SCI da linha e coluna permitidas
//================================================================
//__global__ void calcularLinhaPermitidaKernel(float *dev_matrizSuperior,float *dev_matrizInferior, float ep, int linhaPerm, int totalColunas)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//	//calcular o indice efetivo de calculo
//	i = linhaPerm * totalColunas + i;
//	dev_matrizInferior[i] = dev_matrizSuperior[i] * (1/ep);
//
// }
//
// void SimplexGPUKernels::executarCalculoLinhaPermitida(int numThreads, float *dev_matrizSuperior, float *dev_matrizInferior, float ep, int linhaPerm, int totalColunas)
// {
//
//	 //verificar maneira de adaptar esse controle de forma dinamica
//	int threadsPerBlock = 256;
//	int blocksPerGrid = 1;
//
//	if(threadsPerBlock > numThreads)
//		threadsPerBlock = numThreads;
//	else
//		blocksPerGrid =(numThreads + threadsPerBlock - 1) / threadsPerBlock;
//
//
//	calcularLinhaPermitidaKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_matrizSuperior, dev_matrizInferior, ep, linhaPerm, totalColunas);
//	cudaDeviceSynchronize();
// }
//
//
//
// __global__ void calcularColunaPermitidaKernel(float *dev_matrizSuperior,float *dev_matrizInferior, float ep, int colunaPerm, int totalColunas)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//	//calcular o indice efetivo de calculo
//	i = totalColunas * i + colunaPerm;
//	dev_matrizInferior[i] = dev_matrizSuperior[i] * (-1/ep);
//
// }
//
// void SimplexGPUKernels::executarCalculoColunaPermitida(int numThreads, float *dev_matrizSuperior, float *dev_matrizInferior, float ep, int colunaPerm, int totalColunas)
// {
//	//verificar maneira de adaptar esse controle de forma dinamica
//	int threadsPerBlock = 256;
//	int blocksPerGrid = 1;
//
//	if(threadsPerBlock > numThreads)
//		threadsPerBlock = numThreads;
//	else
//		blocksPerGrid =(numThreads + threadsPerBlock - 1) / threadsPerBlock;
//
//
//	calcularColunaPermitidaKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_matrizSuperior, dev_matrizInferior, ep, colunaPerm, totalColunas);
//	cudaDeviceSynchronize();
// }


__global__ void calcularLinhaColunaPermitidaKernel(float *dev_matrizSuperior, float *dev_matrizInferior, float ep, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  //calcular o indice efetivo de calculo
  int iColunaPerm = totalColunas * i + colunaPerm;
  int iLinhaPerm = linhaPerm * totalColunas + i;

  if (i < totalLinhas)
    //calcula da coluna permissivel
    dev_matrizInferior[iColunaPerm] = dev_matrizSuperior[iColunaPerm] * (-1 / ep);

  if (i < totalColunas)
    //calculo da linha permissivel
    dev_matrizInferior[iLinhaPerm] = dev_matrizSuperior[iLinhaPerm] * (1 / ep);

}

void SimplexGPUKernels::executarCalculoLinhaColunaPermitida(int numThreads, float *dev_matrizSuperior, float *dev_matrizInferior, float ep, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  calcularLinhaColunaPermitidaKernel << <blocksPerGrid, threadsPerBlock >> >(dev_matrizSuperior, dev_matrizInferior, ep, linhaPerm, colunaPerm, totalColunas, totalLinhas);
  cudaDeviceSynchronize();
}


__global__ void calcularQuadro(float *dev_matrizSuperior, float *dev_matrizInferior, int linhaPerm, int colunaPerm, int totalColunas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  //ignorar valores que ja foram preenchidos na matriz inferior
  if (!__isnanf(dev_matrizInferior[i]))
    return;

  //calcular o indice efetivo de calculo
  int iLinhaSCS = linhaPerm * totalColunas + (i % totalColunas);
  int iColunaSCI = ((int)(i / totalColunas)) * totalColunas + colunaPerm;

  //calcular valor efetivo do elemento
  dev_matrizInferior[i] = dev_matrizSuperior[iLinhaSCS] * dev_matrizInferior[iColunaSCI];

  //realizar soma dos elementos entre SCI e SCS, gerando parte do novo quadro
  dev_matrizSuperior[i] = dev_matrizInferior[i] + dev_matrizSuperior[i];

}

void SimplexGPUKernels::executarCalculoQuadro(int numThreads, float *dev_matrizSuperior, float *dev_matrizInferior, int linhaPerm, int colunaPerm, int totalColunas)
{
  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  calcularQuadro << <blocksPerGrid, threadsPerBlock >> >(dev_matrizSuperior, dev_matrizInferior, linhaPerm, colunaPerm, totalColunas);
  cudaDeviceSynchronize();
}


__global__ void transporLinhaColunaPermitidaKernel(float *dev_matrizSuperior, float *dev_matrizInferior, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  //calcular o indice efetivo de calculo
  int iColunaPerm = totalColunas * i + colunaPerm;
  int iLinhaPerm = linhaPerm * totalColunas + i;

  if (i < totalLinhas)
    dev_matrizSuperior[iColunaPerm] = dev_matrizInferior[iColunaPerm];

  if (i < totalColunas)
    dev_matrizSuperior[iLinhaPerm] = dev_matrizInferior[iLinhaPerm];
}

void SimplexGPUKernels::executarTransporLinhaColunaPermitida(int numThreads, float *dev_matrizSuperior, float *dev_matrizInferior, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{

  //verificar maneira de adaptar esse controle de forma dinamica
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;

  if (threadsPerBlock > numThreads)
    threadsPerBlock = numThreads;
  else
    blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  transporLinhaColunaPermitidaKernel << <blocksPerGrid, threadsPerBlock >> >(dev_matrizSuperior, dev_matrizInferior, linhaPerm, colunaPerm, totalColunas, totalLinhas);
  cudaDeviceSynchronize();

}