#ifndef __QUADRO_H_INCLUDED__
#define __QUADRO_H_INCLUDED__

#include "FObjetivo.h"

class Quadro{

public:

  ////Construtor que recebe uma funcao objetivo
  Quadro(FObjetivo* func) : func(func){ };

  void buildQuadro();

  int totalLinhas;
  int totalColunas;

  float *matriz;

  //Metodo que exibe o conteudo do quadro no console. APENAS DEBUG
  void toString();


  std::vector<std::string> colHeader;
  std::vector<std::string> rowHeader;

private:

  FObjetivo* func;

  void buildHeaders();

  void buildMatriz();

};

#endif