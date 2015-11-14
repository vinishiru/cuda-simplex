#ifndef __RESTRICAO_H_INCLUDED__
#define __RESTRICAO_H_INCLUDED__

#include "Variavel.h"
#include <unordered_map>
#include <string>

enum Desigualdade { Maior, Menor, MaiorOuIgual, MenorOuIgual, Igual };

using namespace std;

class Restricao{
public:
  Restricao();
  unordered_map <std::string, Variavel> Variaveis;
  Variavel *VariavelBasica;
  float TermoLivre;
  Desigualdade Desigualdade;
};

#endif