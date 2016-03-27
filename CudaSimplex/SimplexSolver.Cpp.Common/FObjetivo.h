#ifndef __FOBJETIVO_H_INCLUDED__
#define __FOBJETIVO_H_INCLUDED__

#include "Variavel.h"
#include "Restricao.h"
#include <map>
#include <string>


enum TipoOtimizacao { NaoDefinido = 0, Maximizar = 1, Minimizar = 2 };

enum TipoNormalizacao { Tradicional = 0, Petr = 1};

using namespace std;

class FObjetivo{

public:
  FObjetivo();
  string NomeProblema;
  string Nome;
  float TermoLivre; //termo livre da funcao objetivo
  map <string, Variavel> Variaveis; //variaveis da funcao objetivo (Nome da variavel, Variavel)
  map <string, Restricao> Restricoes; // array de restricoes (Nome da restricao, Restricao)
  map <string, Variavel*> VariaveisBasicas; //array de variaveis basicas de todas as restricoes lidas
  map <string, Variavel*> VariaveisArtificiais; //array de variaveis artificiais, usado para o simplex tradicional
  TipoOtimizacao DirecaoOtimizacao; //direcao da otimizacao (minimizar ou maximizar)

  //Funcoes para manipulacao
  void addVariavel(string Nome, float Coeficiente); //adiciona uma variavel nova na funcao objetivo
  void addRestricao(string Nome); //Cria uma nova restricao para a funcao objetivo
  //cria uma restricao com nome sequencial, devolvendo no retorno
  string addRestricao();
  //adicionar uma variavel para um determinada restricao ja adicionada
  void addVariavelRestricao(string NomeRestricao, string NomeVariavel, float Coeficiente);
  //seta o valor de desigualdade para uma determinada restricao
  void setDesigualdadeRestricao(string NomeRestricao, Desigualdade Desigualdade);
  //seta o valor para o termo livre de uma determinada restricao
  void setTermoLivreRestricao(string NomeRestricao, float TermoLivre);

  //Funcao para normalizar de acordo com o padrao de funcao.
  //Ao final, a otimizacao sera do tipo minimizacao, restricoes com igualdades e variaveis artificiais estarao agregadas.
  void normalizar(TipoNormalizacao tipo);

private:

  //Funcao para criar uma nova variavel artificial
  Variavel* criarVariavelBasica();

  //Funcao para criar uma nova variavel artificial (simplex tradicional)
  Variavel* criarVariavelArtificial();

  //Funcoes privadas auxiliares para normalizacao da funcao
  void normalizarExtremo(TipoNormalizacao tipo);
  void normalizarFuncaoObj(TipoNormalizacao tipo);
  void normalizarRestricoes(TipoNormalizacao tipo);

};
#endif