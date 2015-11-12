#include "Quadro.h"
#include <iostream>

void Quadro::buildQuadro(){

  func->normalizar();

  /*
  Nesse ponto e determinada a dimensao dos vetores/matrizes que irao conter
  as celulas inferiores e superiores do algoritmo.
  Um ponto interessante seria a utilizacao de um mecanismo para tratar as matrizes
  de forma a aproveitar melhor o consumo de memoria, tendo em vista que grande parte
  dos problemas de otimizacao sao descritos com conjunto de equacoes que dao
  origem a matrizes esparsas.
  */
  int dim = (func->Variaveis.size() + 1) *  (func->VariaveisBasicas.size() + 1);
  this->matrizSuperior = new float[dim];
  this->matrizInferior = new float[dim];

  //Variaveis auxiliares para indicar o total de linhas e colunas no quadro simplex
  this->totalColunas = func->Variaveis.size() + 1;
  this->totalLinhas = func->VariaveisBasicas.size() + 1;

  this->buildHeaders();

  this->buildMatriz();
}

void Quadro::buildHeaders(){

  //Inserir headers de colunas
  this->colHeader.insert(colHeader.begin(), "Bj");

  int i = 1;
  //Para todos os itens nas variaveis da funcao, adicionar os nomes das variaveis
  for (map<string, Variavel>::iterator it = func->Variaveis.begin(); it != func->Variaveis.end(); it++){
    this->colHeader.insert(colHeader.begin() + i, it->second.Nome);
    i++;
  }

  //Inserir headers das linhas
  this->rowHeader.insert(rowHeader.begin(), "F");
  i = 1;
  //Para todos os itens nas variaveis basicas da funcao, adicionar os nomes das variaveis
  for (map<string, Variavel*>::iterator it = func->VariaveisBasicas.begin(); it != func->VariaveisBasicas.end(); it++){
    this->rowHeader.insert(rowHeader.begin() + i, it->second->Nome);
    i++;
  }
}

void Quadro::buildMatriz(){


  this->matrizSuperior[0] = func->TermoLivre;

  //alocar variaveis da funcao
  int i = 1;
  for (map<string, Variavel>::iterator it = func->Variaveis.begin(); it != func->Variaveis.end(); it++){
    this->matrizSuperior[i] = it->second.Coeficiente;
    i++;
  }

  //variavel de iteracao auxiliar
  map<std::string, Variavel>::iterator auxItVariavel;
  //alocar variaveis das restricoess
  for (map<string, Restricao>::iterator itRestricao = func->Restricoes.begin(); itRestricao != func->Restricoes.end(); itRestricao++){

    //alocar termo livre da restricao
    this->matrizSuperior[i] = itRestricao->second.TermoLivre;
    i++;

    //para cada variavel da funcao objetivo, devemos saber seus respectivos valores
    //na descritos na restricao atual
    for (map<string, Variavel>::iterator itVariavel = func->Variaveis.begin(); itVariavel != func->Variaveis.end(); itVariavel++){

      //tentar recuperar a variavel na restricao atual
      auxItVariavel = itRestricao->second.Variaveis.find(itVariavel->first);

      if (auxItVariavel != itRestricao->second.Variaveis.end())
        this->matrizSuperior[i] = auxItVariavel->second.Coeficiente;
      else
        this->matrizSuperior[i] = 0;

      i++;

      //cout << i << endl;

    }

  }//fim for
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
      cout << this->matrizSuperior[(this->colHeader.size() * linha) + coluna] << "\t";
      coluna++;
    } while (coluna % this->colHeader.size() != 0);

    //contar linha
    linha++;
    coluna = 0;
    cout << endl;
  }

  cout << endl;
}