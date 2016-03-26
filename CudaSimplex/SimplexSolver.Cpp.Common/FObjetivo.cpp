#include "stdafx.h"

FObjetivo::FObjetivo(){
  this->TermoLivre = 0.0;
  this->DirecaoOtimizacao = Minimizar;
}

void FObjetivo::addVariavel(string Nome, float Coeficiente){
  Variavel auxVar;
  auxVar.Nome = Nome;
  auxVar.Coeficiente = Coeficiente;

  this->Variaveis.insert(pair<string, Variavel>(Nome, auxVar));
}

string FObjetivo::addRestricao(){

  string restName = "REST_";
  int countRest;
  Restricao auxRest;

  countRest = this->Restricoes.size();
  restName += to_string(static_cast<long long>(countRest));
  this->Restricoes.insert(pair <string, Restricao>(restName, auxRest));

  return restName;
}

void FObjetivo::addRestricao(string Nome){
  Restricao auxRest;
  this->Restricoes.insert(pair<string, Restricao>(Nome, auxRest));
}

void FObjetivo::addVariavelRestricao(string NomeRestricao, string NomeVariavel, float Coeficiente){
  Restricao *rest;
  Variavel auxVar;
  auxVar.Nome = NomeVariavel;
  auxVar.Coeficiente = Coeficiente;
  rest = &this->Restricoes[NomeRestricao];
  rest->Variaveis.insert(pair<string, Variavel>(NomeVariavel, auxVar));
}

void FObjetivo::setDesigualdadeRestricao(string NomeRestricao, Desigualdade Desigualdade){
  Restricao *rest;
  rest = &this->Restricoes[NomeRestricao];
  rest->Desigualdade = Desigualdade;
}

void FObjetivo::setTermoLivreRestricao(string NomeRestricao, float TermoLivre){

  Restricao *rest;
  rest = &this->Restricoes[NomeRestricao];
  rest->TermoLivre = TermoLivre;
}

Variavel* FObjetivo::criarVariavelBasica(){

  Variavel *varBasica = new Variavel;

  varBasica->Nome = "VB_" + to_string(static_cast<long long>(this->VariaveisBasicas.size()));
  this->VariaveisBasicas.insert(pair<string, Variavel*>(varBasica->Nome, varBasica));

  return varBasica;
}

Variavel* FObjetivo::criarVariavelArtificial(){

  Variavel *varArtificial = new Variavel;

  varArtificial->Nome = "VA_" + to_string(static_cast<long long>(this->VariaveisArtificiais.size()));
  this->VariaveisArtificiais.insert(pair<string, Variavel*>(varArtificial->Nome, varArtificial));

  return varArtificial;
}

void FObjetivo::normalizar(){
  this->normalizarExtremo();
  //this->normalizarFuncaoObj();
  this->normalizarRestricoes();
}

void FObjetivo::normalizarExtremo(){

  //Se a funcao for do tipo Maximizar
  if (this->DirecaoOtimizacao == Maximizar){
    //Inverter sinal das variaveis da funcao objetivo
    for (map<string, Variavel>::iterator it = this->Variaveis.begin(); it != this->Variaveis.end(); it++){
      it->second.Coeficiente = it->second.Coeficiente * (-1);
    }
  }
}

void FObjetivo::normalizarFuncaoObj(){
  for (map<string, Variavel>::iterator it = this->Variaveis.begin(); it != this->Variaveis.end(); it++){
    it->second.Coeficiente = it->second.Coeficiente * (-1);
  }
}

void FObjetivo::normalizarRestricoes(){

  Variavel *varBasica;

  for (map<string, Restricao>::iterator it = this->Restricoes.begin(); it != this->Restricoes.end(); it++){

    varBasica = this->criarVariavelBasica();

    //se a desigualdade for uma relacao menor/menor ou igual,
    //temos que adicionar uma variavel de folga
    if (it->second.Desigualdade == MenorOuIgual || it->second.Desigualdade == Menor || it->second.Desigualdade == Igual){

      varBasica->Coeficiente = 1;
      it->second.VariavelBasica = varBasica;

    }
    //se a desigualdade for maior/maior ou igual,
    //temos que adicionar uma variavel de excesso
    else if (it->second.Desigualdade == MaiorOuIgual || it->second.Desigualdade == Maior){

      varBasica->Coeficiente = -1;
      it->second.VariavelBasica = varBasica;

      //inverter termo livre
      if (it->second.TermoLivre != 0)
        it->second.TermoLivre = it->second.TermoLivre * (-1);

      //inverter valores dos coeficientes de todas as variaveis da restricao
      for (map<string, Variavel>::iterator it2 = it->second.Variaveis.begin(); it2 != it->second.Variaveis.end(); it2++){
        it2->second.Coeficiente = it2->second.Coeficiente * (-1);
      }

    }
    else{
      throw 20;
    }

  }//fim for


}