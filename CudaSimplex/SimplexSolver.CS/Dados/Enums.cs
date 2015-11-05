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
    Maior,
    Menor,
    MenorOuIgual,
    MaiorOuIgual,
    Igual

  }

  public enum StatusSimplex
  {
    SolucaoOtima,
    SolucaoIlimitada,
    SolucaoImpossivel,
    PrimeiraEtapa,
    SegundaEtapa,
    AlgoritmoTroca,
  };

}
