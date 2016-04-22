#include "stdafx.h"

SimplexGPU::SimplexGPU(){

  //inicializar CUDA caso nao esteja inicializado ainda
  if (!this->cudaInitialized){
    this->initCuda();
  }

  this->matrizAlocada = false;
}

void SimplexGPU::initCuda(){

  cudaError_t cudaStatus;

  //inicializar configuracao de device da GPU
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    throw - 1;
  }

  this->cudaInitialized = true;

}

void SimplexGPU::alocarQuadroDevice(Quadro *quadro){

  if (this->matrizAlocada)
    return;

  size_t mallocSize = quadro->totalColunas * quadro->totalLinhas * sizeof(float);

  cudaError_t cudaStatus;

  //alocar matriz superior no device
  cudaStatus = cudaMalloc((void**)&dev_matriz, mallocSize);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizSuperior cudaMalloc failed!");
    throw - 1;
  }
  //copiar matrizes do simplex no device
  cudaStatus = cudaMemcpy(dev_matriz, quadro->matriz, mallocSize, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizSuperior cudaMemcpy failed!");
    throw - 1;
  }

  this->matrizAlocada = true;

}

float* SimplexGPU::calcularQuocientes(Quadro *quadro, int colunaPerm){

  cudaError_t cudaStatus;
  float *vetorQuocientes = new float[quadro->totalLinhas]; //vetor no host
  float *dev_vetorQuocientes = 0; //vetor no device
  size_t mallocSize = quadro->totalLinhas * sizeof(float);

  //solicitar alocacao da matriz superior do quadro
  this->alocarQuadroDevice(quadro);

  //alocar vetor resultado no dispositivo
  cudaStatus = cudaMalloc((void**)&dev_vetorQuocientes, mallocSize);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_vetorQuocientes cudaMalloc failed!");
    throw - 1;
  }

  //executar calculo de quocientes
  SimplexGPUKernels::executarCalcularQuocientesKernel(quadro->totalLinhas - 1, dev_vetorQuocientes, this->dev_matriz, colunaPerm, quadro->totalColunas, quadro->totalLinhas);

  //copiar resultado do calculo do vetor de quocientes do device para o host
  cudaStatus = cudaMemcpy(vetorQuocientes, dev_vetorQuocientes, mallocSize, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    throw - 1;
  }

  //limpar matriz de vetor de quocientes
  cudaStatus = cudaFree(dev_vetorQuocientes);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizInferior cudaFree failed!");
    throw - 1;
  }

  return vetorQuocientes;
}


void SimplexGPU::copiarLinhaPermissivel(Quadro *quadro, int linhaPerm){


  size_t mallocSize = quadro->totalColunas * sizeof(float);

  int copyIndex = quadro->totalColunas * linhaPerm;

  cudaError_t cudaStatus;

  //copiar a linha atualizada na matriz do device
  cudaStatus = cudaMemcpy(dev_matriz + copyIndex, quadro->matriz + copyIndex, mallocSize, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizSuperior cudaMemcpy failed!");
    throw - 1;
  }

}

void SimplexGPU::atualizarQuadro(Quadro *quadro, int colunaPerm, int linhaPerm){

  cudaError_t cudaStatus;

  //copiar a linha atualizada do elemento permissivel no device
  this->copiarLinhaPermissivel(quadro, linhaPerm);

  //tamanho total de memoria alocada que sera lida do device e gravada no host
  size_t mallocSize = quadro->totalColunas * quadro->totalLinhas * sizeof(float);

  //para o algoritmo de troca, precisaremos dos valores que irao anular
  //os elementos da coluna correspondente a do elemento permissivel,
  //dessa forma, lancaremos 1 thread por linha para calcular esse fator
  //de anulacao, e calcular os elemetos da respectiva linha computada
  SimplexGPUKernels::executarCalculoAlgoritmoTroca(quadro->totalLinhas, this->dev_matriz, linhaPerm, colunaPerm, quadro->totalColunas);

  //copiar quadro calculado device -> host
  cudaStatus = cudaMemcpy(quadro->matriz, dev_matriz, mallocSize, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    throw - 1;
  }

}

