#ifndef __RESTRICAO_H_INCLUDED__
#define __RESTRICAO_H_INCLUDED__

#include "Variavel.h"
#include <map>
#include <string>

enum Desigualdade { Maior, Menor, MaiorOuIgual, MenorOuIgual, Igual };

using namespace std;

class Restricao{
public:
  Restricao();
  map <std::string, Variavel> Variaveis;
  Variavel *VariavelBasica;
  Variavel *VariavelArtificial; //simplex tradicional
  float TermoLivre;
  Desigualdade Desigualdade;
};

#endif