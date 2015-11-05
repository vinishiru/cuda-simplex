using SimplexSolver.CS.Dados;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.CS.Interfaces
{
  public interface ILPReader
  {
    FObjetivo LerFuncaoObjetivo();
  }
}
