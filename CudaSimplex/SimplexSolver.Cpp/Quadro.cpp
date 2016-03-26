#include "stdafx.h"

void Quadro::buildQuadro(){

  func->normalizar();

  /*
  Determinar a dimensao da matriz a ser utilizada no desenvolvimento
  do algoritmo Simplex tradicional.
  A dimensao da matriz sera:
  TotalLinhas = Total de Restricoes + 1 (FO)
  TotalColunas = Total de variaveis FO + Total de variaveis basicas + 2 ( P e Termo Livre )
  */
  this->totalLinhas = func->Restricoes.size() + 1;
  this->totalColunas = func->Variaveis.size() + func->VariaveisBasicas.size() + 2;
  int dim = this->totalLinhas * this->totalColunas;

  this->matriz = new float[dim];

  //construir headers
  this->buildHeaders();

  //construir matriz do quadro
  this->buildMatriz();

}

void Quadro::buildHeaders(){

  int i = 0;

  //Para todos os itens nas variaveis da funcao, adicionar os nomes das variaveis
  for (map<string, Variavel>::iterator it = func->Variaveis.begin(); it != func->Variaveis.end(); it++){
    this->colHeader.insert(colHeader.begin() + i, it->second.Nome);
    i++;
  }

  //adicionar os itens das variaveis basicas
  //Percorrer todos as restricoes e adicionar as respectivas variaveis basicas
  for (map<string, Variavel*>::iterator it = func->VariaveisBasicas.begin(); it != func->VariaveisBasicas.end(); it++){
    this->colHeader.insert(colHeader.begin() + i, it->second->Nome);
    i++;
  }

  //inserir header final - P
  this->colHeader.insert(colHeader.begin() + i, "P");
  i++;
  //inserir header final - Bj
  this->colHeader.insert(colHeader.begin() + i, "Bj");
  i = 0;

  for (map<string, Variavel*>::iterator it = func->VariaveisBasicas.begin(); it != func->VariaveisBasicas.end(); it++){
    this->rowHeader.insert(rowHeader.begin() + i, it->second->Nome);
    i++;
  }

  //inserir header final - P
  this->rowHeader.insert(rowHeader.begin() + i, "P");

}


void Quadro::buildMatriz(){

  int i = 0;


  //variavel de iteracao auxiliar
  map<std::string, Variavel>::iterator auxItVariavel;

  //atribuir restricoes e suas respectivas variaveis basicas, P, e o termo livre
  for (map<string, Restricao>::iterator itRestricao = func->Restricoes.begin(); itRestricao != func->Restricoes.end(); itRestricao++)
  {

    //para cada variavel da funcao objetivo, devemos saber seus respectivos valores
    //na descritos na restricao atual
    for (map<string, Variavel>::iterator itVariavel = func->Variaveis.begin(); itVariavel != func->Variaveis.end(); itVariavel++){

      //tentar recuperar a variavel na restricao atual
      auxItVariavel = itRestricao->second.Variaveis.find(itVariavel->first);

      if (auxItVariavel != itRestricao->second.Variaveis.end())
        this->matriz[i] = auxItVariavel->second.Coeficiente;
      else
        this->matriz[i] = 0;

      i++;
    }

    //percorrer o array de variaveis basicas e atribuir as que possuem valor na restricao atual
    //a matriz do quadro
    for (map<string, Variavel*>::iterator itVariavel = func->VariaveisBasicas.begin(); itVariavel != func->VariaveisBasicas.end(); itVariavel++){

      //verificar se a variavel basica possui valor na restricao atual
      if (itRestricao->second.VariavelBasica->Nome.compare(itVariavel->second->Nome) == 0)
        this->matriz[i] = 1;
      else
        this->matriz[i] = 0;

      i++;
    }

    //setar valor da coluna P = 0
    this->matriz[i] = 0;
    i++;
    //setar valor do termo livre da restricao atual na ultima coluna
    this->matriz[i] = itRestricao->second.TermoLivre;
    i++;
  }

  //setar valores da funcao objetivo na ultima linha do quadro
  for (map<string, Variavel>::iterator itVariavel = func->Variaveis.begin(); itVariavel != func->Variaveis.end(); itVariavel++){
    this->matriz[i] = itVariavel->second.Coeficiente;
    i++;
  }

  //setar valores das variaveis basicas iguais a zero na ultima linha
  for (map<string, Variavel*>::iterator itVariavel = func->VariaveisBasicas.begin(); itVariavel != func->VariaveisBasicas.end(); itVariavel++){
    this->matriz[i] = 0;
    i++;
  }

  //setar valor P = 1
  this->matriz[i] = 1;
  i++;
  //setar valor de custo inicial = 0
  this->matriz[i] = 0;

}


void Quadro::toString(){

  cout << endl;
  cout << "Conteudo do Quadro Simplex" << endl << endl;
  cout << endl;

  cout << "\t";

  //escrever os cabecalhos inicialmente
  for (std::vector<std::string>::iterator it = this->colHeader.begin(); it != this->colHeader.end(); it++){
    cout << *it << "\t";
  }

  cout << endl;

  int coluna = 0;
  int linha = 0;

  //escrever o inicio de cada linha seguido do 
  for (std::vector<std::string>::iterator it = this->rowHeader.begin(); it != this->rowHeader.end(); it++){
    //escrever header da linha
    cout << *it << "\t";

    //escrever elementos da linha
    do
    {
      cout << this->matriz[(this->colHeader.size() * linha) + coluna] << "\t";
      coluna++;
    } while (coluna % this->colHeader.size() != 0);

    //escrever ultimo elemento

    //contar linha
    linha++;
    coluna = 0;
    cout << endl;
  }

  cout << endl;
}