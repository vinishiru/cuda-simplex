#include "SimplexGPU.h"

SimplexGPU::SimplexGPU(){

  //inicializar CUDA caso nao esteja inicializado ainda
  if (!this->cudaInitialized){
    this->initCuda();
  }

  this->matrizAlocada = false;
  this->linhaColunaPermAlocadas = false;
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
  cudaStatus = cudaMalloc((void**)&dev_matrizSuperior, mallocSize);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizSuperior cudaMalloc failed!");
    throw - 1;
  }
  //copiar matrizes do simplex no device
  cudaStatus = cudaMemcpy(dev_matrizSuperior, quadro->matrizSuperior, mallocSize, cudaMemcpyHostToDevice);
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
  SimplexGPUKernels::executarCalcularQuocientesKernel(quadro->totalLinhas, dev_vetorQuocientes, this->dev_matrizSuperior, colunaPerm, quadro->totalColunas, quadro->totalLinhas);

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


void SimplexGPU::alocarLinhaColunaPermissiveis(Quadro *quadro, int colunaPerm, int linhaPerm){


  int maiorDim = quadro->totalLinhas > quadro->totalColunas ? quadro->totalLinhas : quadro->totalColunas;

  if (!linhaColunaPermAlocadas){
    //alocar espaco da linha e coluna permissiveis no device

    cudaError_t cudaStatus;
    size_t mallocSize = quadro->totalColunas * sizeof(float);

    cudaStatus = cudaMalloc((void**)&dev_linhaPerm, mallocSize);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "dev_linhaPerm cudaMalloc failed!");
      throw - 1;
    }

    mallocSize = quadro->totalLinhas * sizeof(float);
    cudaStatus = cudaMalloc((void**)&dev_colunaPerm, mallocSize);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "dev_colunaPerm cudaMalloc failed!");
      throw - 1;
    }
  }

  //funcao para gravar em memoria os valores da linha e coluna permitidas
  //usadas para efetuar o algoritmo de troca
  SimplexGPUKernels::executarCopiarLinhaColunaPermissiveis(maiorDim, dev_matrizSuperior, dev_linhaPerm, dev_colunaPerm, linhaPerm, colunaPerm, quadro->totalColunas, quadro->totalLinhas);

}

void SimplexGPU::atualizarQuadro(Quadro *quadro, int colunaPerm, int linhaPerm){

  cudaError_t cudaStatus;

  //tamanho total de memoria alocada que sera lida do device e gravada no host
  size_t mallocSize = quadro->totalColunas * quadro->totalLinhas * sizeof(float);

  //posicao do elemento permissivel
  int epIndex = linhaPerm * quadro->totalColunas + colunaPerm;

  //elemento permitido (EP)
  float ep = quadro->matrizSuperior[epIndex];

  this->alocarLinhaColunaPermissiveis(quadro, colunaPerm, linhaPerm);

  //metodo otimizado para calcular as SCIs
  SimplexGPUKernels::executarCalculoAlgoritmoTroca(quadro->totalLinhas * quadro->totalColunas, ep, this->dev_matrizSuperior, this->dev_linhaPerm, this->dev_colunaPerm, linhaPerm, colunaPerm, quadro->totalColunas);

  //copiar quadro calculado device -> host
  cudaStatus = cudaMemcpy(quadro->matrizSuperior, dev_matrizSuperior, mallocSize, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    throw - 1;
  }

}

