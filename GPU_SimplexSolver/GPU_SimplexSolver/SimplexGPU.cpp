#include "SimplexGPU.h"

SimplexGPU::SimplexGPU(){

  //inicializar CUDA caso nao esteja inicializado ainda
  if (!this->cudaInitialized){
    this->initCuda();
  }
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

  float *tempDev_matrizInferior = 0;
  size_t mallocSize = quadro->totalColunas * quadro->totalLinhas * sizeof(float);

  cudaError_t cudaStatus;

  //alocar matriz superior no device
  cudaStatus = cudaMalloc((void**)&dev_matrizSuperior, mallocSize);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizSuperior cudaMalloc failed!");
    throw - 1;
  }

  //alocar matriz inferior no device
  cudaStatus = cudaMalloc((void**)&dev_matrizInferior, mallocSize);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizInferior cudaMalloc failed!");
    throw - 1;
  }

  //copiar matrizes do simplex no device
  cudaStatus = cudaMemcpy(dev_matrizSuperior, quadro->matrizSuperior, mallocSize, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizSuperior cudaMemcpy failed!");
    throw - 1;
  }

  //instanciar vetor temporario da matriz inferior
  tempDev_matrizInferior = new float[quadro->totalColunas * quadro->totalLinhas];
  //preencher com valor especial NAN
  std::fill_n(tempDev_matrizInferior, quadro->totalColunas * quadro->totalLinhas, numeric_limits<float>::quiet_NaN());
  //copiar matriz inferior com valor especial
  cudaStatus = cudaMemcpy(dev_matrizInferior, tempDev_matrizInferior, mallocSize, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizInferior cudaMemcpy failed!");
    throw - 1;
  }
  delete tempDev_matrizInferior;
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

void SimplexGPU::atualizarQuadro(Quadro *quadro, int colunaPerm, int linhaPerm){

  cudaError_t cudaStatus;

  //tamanho total de memoria alocada que sera lida do device e gravada no host
  size_t mallocSize = quadro->totalColunas * quadro->totalLinhas * sizeof(float);

  int maiorDim = quadro->totalLinhas > quadro->totalColunas ? quadro->totalLinhas : quadro->totalColunas;

  //posicao do elemento permissivel
  int epIndex = linhaPerm * quadro->totalColunas + colunaPerm;

  //elemento permitido (EP)
  float ep = quadro->matrizSuperior[epIndex];

  //multiplicar toda a linha pelo EP inverso encontrado e
  //multiplicar toda coluna pelo -EP inverso
  SimplexGPUKernels::executarCalculoLinhaColunaPermitida(maiorDim, this->dev_matrizSuperior, this->dev_matrizInferior, ep, linhaPerm, colunaPerm, quadro->totalColunas, quadro->totalLinhas);

  //as outras posicoes da matrizInferior serao calculadas da seguinte forma:
  //Elemento = SCI da coluna permitida * SCS da linha permitida
  //correspondentes a do elemento
  SimplexGPUKernels::executarCalculoQuadro(quadro->totalLinhas * quadro->totalColunas, this->dev_matrizSuperior, this->dev_matrizInferior, linhaPerm, colunaPerm, quadro->totalColunas);

  //transpor os elementos das SCI da linha e coluna permitidas para as suas SCS correspondentes
  SimplexGPUKernels::executarTransporLinhaColunaPermitida(maiorDim, this->dev_matrizSuperior, this->dev_matrizInferior, linhaPerm, colunaPerm, quadro->totalColunas, quadro->totalLinhas);

  //copiar quadro device -> host
  cudaStatus = cudaMemcpy(quadro->matrizSuperior, dev_matrizSuperior, mallocSize, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    throw - 1;
  }

  //setar o inverso do elemento permitido na sua respectiva SCS
  quadro->matrizSuperior[epIndex] = 1 / ep;

  //desalocar matriz superior e inferior do device
  cudaStatus = cudaFree(this->dev_matrizSuperior);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizInferior cudaFree failed!");
    throw - 1;
  }
  cudaStatus = cudaFree(this->dev_matrizInferior);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_matrizInferior cudaFree failed!");
    throw - 1;
  }

}

