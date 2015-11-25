using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.CS.Dados
{
  public enum Extremo
  {
    Minimizar,
    Maximizar
  }

  public enum Desigualdade
  {
    Maior = 0,
    Menor = 1,
    MenorOuIgual = 2,
    MaiorOuIgual = 3,
    Igual = 4

  }

  public enum StatusSimplex
  {
    SolucaoOtima = 0,
    SolucaoIlimitada = 1,
    SolucaoImpossivel = 2,
    PrimeiraEtapa = 3,
    SegundaEtapa = 4,
    AlgoritmoTroca = 5,
  };

}
