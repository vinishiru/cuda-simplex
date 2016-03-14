#ifndef __QUADRO_H_INCLUDED__
#define __QUADRO_H_INCLUDED__

#include "FObjetivo.h"
#include <exception>
#include <string>
#include <vector>

class Quadro{

public:
	//Construtor que recebe uma funcao objetivo normalizada.
	Quadro(FObjetivo* func) : func(func){ };
	
	//Funcao para construir o quadro (tableau) Simplex
	void buildQuadro();

	//Metodo que exibe o conteudo do quadro no console. APENAS DEBUG
	void toString();

	int totalLinhas;
	int totalColunas;

	//Matrizes do quadro
	float *matrizSuperior;
	float *matrizInferior;

	std::vector<std::string> colHeader;
	std::vector<std::string> rowHeader;

private:
	//Funcao objetivo recebida
	FObjetivo* func;

	//Funcao para construir os cabecalhos para linha e colunas do quadro, com os nomes
	//das variaveis nao basicas e basicas
	void buildHeaders();
	//Metodo para construir as matrizes
	//Subcelula Superior (SCS) e Subcelula Inferior (SCI)
	void buildMatriz();

	
};

#endif