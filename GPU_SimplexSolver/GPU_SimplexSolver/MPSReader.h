#ifndef __MPSREADER_H_INCLUDED
#define __MPSREADER_H_INCLUDED

#include "FObjetivo.h"
#include "FileReader.h"
#include "ILPReader.h"
#include <string>

class MPSReader : public ILPReader {

public:
  /**Faz todo o processamento de:
  - Adicionar uma funcao objetivo, variaveis e termo livre;
  - Adicionar N restricoes para a funcao objetivo;
  */
  /*void readFObjetivo(FObjetivo *func, std::string MPSLocation);*/

  //Construtor
  MPSReader(std::string mpsPath);
  //Destrutor
  //~MPSReader();

  virtual FObjetivo* LerFuncaoObjetivo();

private:

  FileReader *fileReader;
  FObjetivo *funcao;
  std::string line;

  void readNAME(); //funcao que define o proximo token como NAME da funcao
  void readROWS(); //funcao que le 2 tokens para determinar a desigualdade das funcoes e retricoes
  void readCOLUMNS(); //funcao que le ate 5 tokens para atribuir variaveis
  void readRHS(); //funcao que le ate 5 tokens para determinar termos independentes das funcoes e retricoes
  void readBOUNDS(); //funcao que le 4 tokens para determinar limites (restricoes adicionais para variaveis)

};

#endif