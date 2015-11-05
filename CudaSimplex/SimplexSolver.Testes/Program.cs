using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.Testes
{
  class Program
  {
    static void Main(string[] args)
    {

      testarSimplexSolverCS();

    }

    private static void testarSimplexSolverCS()
    {
      SimplexSolver.CS.SimplexSolverCPU solver = new CS.SimplexSolverCPU();
      SimplexSolver.CS.Dados.FObjetivo funcao = new CS.Dados.FObjetivo();


      solver.Otimizar(funcao);

    }
  }
}
