using SimplexSolver.CS.Dados;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.CS.Interfaces
{
  interface ISimplexSolver
  {

    TimeSpan TempoLeituraFuncao();

    TimeSpan TempoNormalizacao();

    TimeSpan TempoOtimizacao();

    StatusSimplex RecuperarStatus();

    void Otimizar(ILPReader reader, Extremo extremo);

    void Otimizar(FObjetivo funcao, Extremo extremo);

  }
}
