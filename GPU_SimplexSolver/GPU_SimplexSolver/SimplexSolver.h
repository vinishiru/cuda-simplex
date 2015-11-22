#ifndef __SIMPLEX_SOLVER_H_INCLUDED__
#define __SIMPLEX_SOLVER_H_INCLUDED__

#include "FObjetivo.h"
#include "Quadro.h"
#include "SimplexGPU.h"
#include "ILPReader.h"
#include "Stopwatch.h"
#include <vector>
#include <exception>
#include <iostream>

enum StatusSimplex
{
  SolucaoOtima = 0,
  SolucaoIlimitada = 1,
  SolucaoImpossivel = 2,
  PrimeiraEtapa = 3,
  SegundaEtapa = 4,
  AlgoritmoTroca = 5,
};

/*
Classe SimplexSolver
Responsavel pela otimizacao da funcao objetivo passada
via parametro no construtor.
Devera analisar as capacidades de execucao da maquina do cliente
seguindo os seguintes procedimentos:
- Verificar disponibilidade de um dispositivo compativel com CUDA.
- Verificar disponibilidade de execucao usando bibliotecas paralelas OpenMP.

A classe encapsulara os recursos a serem utilizados na otimizacao do problema
de programacao linear, tentando utilizar os recursos disponiveis da melhor
maneira possivel, favorecendo a performance do procedimento Simplex.
**/
class SimplexSolver {

public:

  //Funcao que realizara todo o trabalho de otimizacao da funcao objetivo
  void otimizar(ILPReader* reader);

  void otimizar(FObjetivo* func);

  double tempoLeitura();
  double tempoNormalizacao();
  double tempoOtimizacao();

  StatusSimplex statusFinal();

private:

  Stopwatch swLeitura;
  Stopwatch swNormalizacao;
  Stopwatch swOtimizacao;

  SimplexGPU simplexGPU;
  vector<StatusSimplex> historico;
  StatusSimplex status;
  Quadro *quadro;

  int linhaPerm;
  int colunaPerm;

  StatusSimplex algoritmoPrimeiraEtapa();
  StatusSimplex algoritmoSegundaEtapa();
  StatusSimplex algoritmoTroca();

  void calcularLinhaPermissivel();

};

#endif