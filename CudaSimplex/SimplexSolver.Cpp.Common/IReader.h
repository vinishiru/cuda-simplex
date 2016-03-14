#ifndef __IREADER_H_INCLUDED__
#define __IREADER_H_INCLUDED__

#include "FObjetivo.h" 

class IReader{
public:
	/**Faz todo o processamento de:
	- Adicionar uma funcao objetivo, variaveis e termo livre;
	- Adicionar N restricoes para a funcao objetivo;
	*/
	virtual void readFObjetivo(FObjetivo &funcao);
};

#endif