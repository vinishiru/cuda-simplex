using SimplexSolver.CS.Dados;
using SimplexSolver.CS.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.CS.Classes
{
  public class MPSLPReader : ILPReader
  {

    public FObjetivo Funcao { get; set; }

    private string _path;

    public MPSLPReader(string path)
    {
      _path = path;
    }

    public Dados.FObjetivo LerFuncaoObjetivo()
    {
      throw new NotImplementedException();
    }
  }
}
