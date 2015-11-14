#ifndef __FOBJETIVO_H_INCLUDED__
#define __FOBJETIVO_H_INCLUDED__

#include "Variavel.h"
#include "Restricao.h"
#include <unordered_map>
#include <string>


enum TipoOtimizacao { NaoDefinido = 0, Maximizar = 1, Minimizar = 2 };

using namespace std;

class FObjetivo{

public: 
	FObjetivo();
	string NomeProblema;
	string Nome;
	float TermoLivre; //termo livre da funcao objetivo
  unordered_map <string, Variavel> Variaveis; //variaveis da funcao objetivo (Nome da variavel, Variavel)
  unordered_map <string, Restricao> Restricoes; // array de restricoes (Nome da restricao, Restricao)
  unordered_map <string, Variavel*> VariaveisBasicas; //array de variaveis basicas de todas as restricoes lidas
	TipoOtimizacao DirecaoOtimizacao; //direcao da otimizacao (minimizar ou maximizar)

	//Funcoes para manipulacao
	void addVariavel( string Nome, float Coeficiente); //adiciona uma variavel nova na funcao objetivo
	void addRestricao( string Nome); //Cria uma nova restricao para a funcao objetivo
	//cria uma restricao com nome sequencial, devolvendo no retorno
	string addRestricao();
	//adicionar uma variavel para um determinada restricao ja adicionada
	void addVariavelRestricao( string NomeRestricao, string NomeVariavel, float Coeficiente);
	//seta o valor de desigualdade para uma determinada restricao
	void setDesigualdadeRestricao (string NomeRestricao, Desigualdade Desigualdade);
	//seta o valor para o termo livre de uma determinada restricao
	void setTermoLivreRestricao( string NomeRestricao, float TermoLivre);

	//Funcao para normalizar de acordo com o padrao de funcao.
	//Ao final, a otimizacao sera do tipo minimizacao, restricoes com igualdades e variaveis artificiais estarao agregadas.
	void normalizar();

private:
	//Funcao para criar uma nova variavel artificial
	Variavel* criarVariavelBasica();
	//Funcoes privadas auxiliares para normalizacao da funcao
	void normalizarExtremo();
  void normalizarFuncaoObj();
	void normalizarRestricoes();

};
#endif