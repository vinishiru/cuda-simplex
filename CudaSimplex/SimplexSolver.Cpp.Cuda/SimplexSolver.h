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

  double tempoLeitura();
  double tempoNormalizacao();
  double tempoOtimizacao();
  float valorCusto();

private:

  Stopwatch swLeitura;
  Stopwatch swNormalizacao;
  Stopwatch swOtimizacao;
  Stopwatch swSegPorIteracao;

  Quadro *quadro;

  SimplexGPU simplexGPU;
  StatusSimplex status;

  int linhaPerm;
  int colunaPerm;

  StatusSimplex algoritmoPrimeiraEtapa();
  StatusSimplex algoritmoSegundaEtapa();
  StatusSimplex algoritmoTroca();

  void calcularLinhaPermissivel();

  float recuperarTermoLivreLinha(int linha);

  bool colunaPertenceABase(int coluna);

};

