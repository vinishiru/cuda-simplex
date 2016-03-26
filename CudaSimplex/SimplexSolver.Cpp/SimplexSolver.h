#pragma once

enum StatusSimplex
{
  SolucaoOtima = 0,
  SolucaoIlimitada = 1,
  SolucaoImpossivel = 2,
  PrimeiraEtapa = 3,
  SegundaEtapa = 4,
  AlgoritmoTroca = 5,
};

class SimplexSolver
{
public:

  void otimizar(FObjetivo* func);

private:

  Quadro *quadro;

  int linhaPerm;
  int colunaPerm;

};

