using SimplexSolver.CS.Dados;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.Testes
{
  public class GeradorLP
  {

    class VariavelCusto
    {
      public string Nome { get; set; }
      public int ValorCusto { get; set; }
      public int Coeficiente { get; set; }

    }

    const string PREFIXO_RESTRICAO = "REST_";
    const string PREFIXO_VARIAVEL = "X_";

    List<VariavelCusto> _variaveis;
    FObjetivo _funcaoObj;

    int _qtdVariaveis;
    int _qtdRestricoes;
    int _densidadeMatriz;

    Random _rnd;

    public GeradorLP(int qtdVariaveis, int qtdRestricoes, int densidadeMatriz)
    {
      _qtdRestricoes = qtdRestricoes;
      _qtdVariaveis = qtdVariaveis;
      _densidadeMatriz = densidadeMatriz;

      _rnd = new Random();
      _funcaoObj = new FObjetivo();
      _variaveis = new List<VariavelCusto>();
    }

    public FObjetivo GerarProblemaLP()
    {
      GerarFuncaoObj();
      GerarRestricoes();

      return _funcaoObj;
    }

    private void GerarRestricoes()
    {

      Restricao auxRest = null;
      int custoRest = 0;
      int coeficienteVar = 0;
      for (int i = 0; i < _qtdRestricoes; i++)
      {
        custoRest = 0;
        auxRest = _funcaoObj.AddRestricao(PREFIXO_RESTRICAO + i);
        auxRest.Desigualdade = GerarDesigualdade();

        foreach (var variavel in _variaveis)
        {
          if (GerarBool(_densidadeMatriz))
          {
            coeficienteVar = _rnd.Next(-9, 9);
            auxRest.AddVariavel(variavel.Nome, coeficienteVar);
            custoRest += coeficienteVar * variavel.ValorCusto;
          }
        }
        //calcular termo livre em funcao do custo acumulado para a restricao
        auxRest.TermoLivre = GerarTermoLivreFactivelRestricao(custoRest, auxRest.Desigualdade);
      }
    }

    private void GerarFuncaoObj()
    {
      VariavelCusto auxVar = null;
      _funcaoObj.Nome = string.Format("{0}Var_{1}Rest", _qtdVariaveis, _qtdRestricoes);
      _funcaoObj.NomeProblema = _funcaoObj.Nome;
      _funcaoObj.Extremo = Extremo.Minimizar;

      for (int i = 0; i < _qtdVariaveis; i++)
      {
        auxVar = GerarVariavelCusto(PREFIXO_VARIAVEL + i);
        _variaveis.Add(auxVar);
        _funcaoObj.AddVariavel(auxVar.Nome, auxVar.Coeficiente);
      }
    }

    Boolean GerarBool(int probabilidade)
    {
      return _rnd.Next(0, 100) <= probabilidade;
    }

    private int GerarTermoLivreFactivelRestricao(int custoRest, Desigualdade desigualdade)
    {
      switch (desigualdade)
      {
        case Desigualdade.Menor:
          return _rnd.Next(custoRest + 1, custoRest + _rnd.Next(2, 100));

        case Desigualdade.MenorOuIgual:
          return _rnd.Next(custoRest, custoRest + _rnd.Next(1, 100));

        case Desigualdade.Maior:
          return _rnd.Next(custoRest - _rnd.Next(2, 100), custoRest - 1);

        case Desigualdade.MaiorOuIgual:
          return _rnd.Next(custoRest - _rnd.Next(1, 100), custoRest);

        default:
          throw new ApplicationException("Foi gerado uma desigualdade invalida.");
      }
    }

    Desigualdade GerarDesigualdade()
    {
      return (Desigualdade)Enum.Parse(typeof(Desigualdade), _rnd.Next(0, 3).ToString());
    }

    VariavelCusto GerarVariavelCusto(string nome)
    {
      VariavelCusto varCusto = new VariavelCusto();

      varCusto.Nome = nome;
      varCusto.Coeficiente = _rnd.Next(1, 9);
      varCusto.ValorCusto = _rnd.Next(0, 9);

      return varCusto;
    }

  }
}
