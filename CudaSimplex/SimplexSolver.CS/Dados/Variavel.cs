
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;

namespace SimplexSolver.CS.Dados
{

  public class Variavel
  {

    //Propriedades que compõem uma variável
    public string Nome { get; set; }
    //Nome da variável (ex.: x1, x2, etc...)
    public double Coeficiente { get; set; }
    //Valor do coeficiente multiplicador

  }
}