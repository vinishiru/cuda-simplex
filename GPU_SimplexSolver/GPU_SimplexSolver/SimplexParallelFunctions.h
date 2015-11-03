#ifndef __SIMPLEX_PARALLEL_FUNCTIONS_H_INCLUDED__
#define __SIMPLEX_PARALLEL_FUNCTIONS_H_INCLUDED__

#include "Quadro.h"
/*
Classe abstrata que representa os metodos
potencialmente paralelizaveis do algoritmo Simplex.

Dessa forma, uma classe derivada pode implementar diferentes maneiras
de executar tais funcoes que tem potencial paralelo da melhor
maneira possivel.
*/

class SimplexParallelFunctions {

public:

	//funcao para calcular os quocientes entre os membros livres
	//e a coluna permissivel, a fim de se encontrar o indice da linha
	//permissivel.
	//Retorna um array com o resultado da operacao entre os dois vetores
	virtual float* calcularQuocientes(Quadro* quadro, int colPerm) = 0;

};

#endif