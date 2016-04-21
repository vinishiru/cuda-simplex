#pragma once

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