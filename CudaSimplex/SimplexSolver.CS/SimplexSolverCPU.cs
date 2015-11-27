using SimplexSolver.CS.Dados;
using SimplexSolver.CS.Interfaces;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.CS
{

  public class SimplexSolverCPU : ISimplexSolver
  {

    private Quadro _quadro;

    private int _linhaPerm;
    private int _colunaPerm;

    private int _qtdIteracoes;
    private Stack<StatusSimplex> _historico;
    private StatusSimplex _statusSimplex;

    private Stopwatch _swLeituraFuncao;
    private Stopwatch _swNormalizacao;
    private Stopwatch _swOtimizacao;

    public SimplexSolverCPU()
    {
      _statusSimplex = new StatusSimplex();
      _historico = new Stack<StatusSimplex>();
    }

    public void Otimizar(ILPReader reader, Extremo extremo = Extremo.Minimizar)
    {
      if (reader == null)
        throw new ArgumentException("reader");

      _swLeituraFuncao = Stopwatch.StartNew();
      var funcao = reader.LerFuncaoObjetivo();
      _swLeituraFuncao.Stop();

      Otimizar(funcao, extremo);
    }

    public void Otimizar(FObjetivo funcao, Extremo extremo = Extremo.Minimizar)
    {
      Console.WriteLine();
      Console.WriteLine("Quantidade variaveis: {0} Quantidade restricoes: {1}", funcao.Variaveis.Count, funcao.Restricoes.Count);
      Console.WriteLine();
      funcao.Extremo = extremo;

      if (!funcao.Normalizado)
      {
        _swNormalizacao = Stopwatch.StartNew();
        funcao.normalizar();
        _quadro = new Quadro(funcao);
        _swNormalizacao.Stop();
      }

      _qtdIteracoes = 1;
      _historico.Clear();

      //mostrar quadro apos a troca
      //Console.WriteLine(_quadro.ToString());

      _statusSimplex = StatusSimplex.PrimeiraEtapa;

      _swOtimizacao = Stopwatch.StartNew();

      while (_statusSimplex != StatusSimplex.SolucaoIlimitada &&
        _statusSimplex != StatusSimplex.SolucaoOtima &&
        _statusSimplex != StatusSimplex.SolucaoImpossivel)
      {

        _historico.Push(_statusSimplex);

        switch (_statusSimplex)
        {
          case StatusSimplex.PrimeiraEtapa:
            _statusSimplex = this.AlgoritmoPrimeiraEtapa();
            break;

          case StatusSimplex.SegundaEtapa:
            _statusSimplex = this.AlgoritmoSegundaEtapa();
            break;

          case StatusSimplex.AlgoritmoTroca:
            Console.WriteLine(string.Format("Linha Permissivel: {0}\tColuna Permissivel: {1}\tIteracoes: {2}", _linhaPerm, _colunaPerm, _qtdIteracoes));
            _statusSimplex = this.AlgoritmoTroca();
            _qtdIteracoes++;
            break;
        }
      }
      //fim otimizacao
      _swOtimizacao.Stop();

      _historico.Push(_statusSimplex);

    }

    private StatusSimplex AlgoritmoPrimeiraEtapa()
    {
      bool tlNegativoEncontrado = false;
      bool elemNegativoEncontrado = false;
      int linhaTLNegativo = 0;

      //zerar linha e coluna permissiveis
      _colunaPerm = -1;
      _linhaPerm = -1;

      //procurar por um elemento negativo na coluna dos termos livres,
      //escolhendo o primeiro elemento negativo encontrado
      for (int i = 0; i < _quadro.TotalLinhas(); i++)
      {
        if (_quadro.Matriz[i, 0].ValorSuperior < 0)
        {
          linhaTLNegativo = i;
          tlNegativoEncontrado = true;
          break; //sair do for
        }
      }

      //Se o elemento negativo nao existe, ir para segunda etapa do
      //algoritmo
      if (!tlNegativoEncontrado)
        return StatusSimplex.SegundaEtapa;

      //procurar por elemento negativo na linha do termo livre negativo,
      //escolhendo o primeiro elemento negativo encontrado
      for (int j = 1; j < _quadro.TotalColunas(); j++)
      {
        if (_quadro.Matriz[linhaTLNegativo, j].ValorSuperior < 0)
        {
          //marcar coluna permissivel
          _colunaPerm = j;
          elemNegativoEncontrado = true;
          break; //sair do for
        }
      }

      //se nao existir elemento negativo na mesma linha do termo livre
      //negativo, entao nao existe solucao permissivel
      if (!elemNegativoEncontrado)
        return StatusSimplex.SolucaoImpossivel;

      //calcular linha permissivel
      this.CalcularLinhaPermissivel();

      return StatusSimplex.AlgoritmoTroca;
    }

    private StatusSimplex AlgoritmoSegundaEtapa()
    {
      bool colunaFPositivoEncontrado = false;
      bool elePositivoEncontrado = false;

      //zerar linha e coluna permissiveis
      _colunaPerm = -1;
      _linhaPerm = -1;

      //na linha F(X) procuramos por um elemento positivo,
      //desconsiderando o termo livre
      for (int i = 1; i < _quadro.TotalColunas(); i++)
      {
        if (_quadro.Matriz[0, i].ValorSuperior > 0)
        {
          _colunaPerm = i;
          colunaFPositivoEncontrado = true;
          break;
        }
      }

      //se nao encontrarmos elemento positivo na linha F(X),
      //significa que a solucao atual e otima
      if (!colunaFPositivoEncontrado)
        return StatusSimplex.SolucaoOtima;

      //usando a coluna F(x) de elemento positivo,
      //tentar encontrar um elemento positivo nas linhas abaixo,
      //da linha F(x), serve apenas para verificar se a solucao nao e ilimitada
      for (int i = 1; i < _quadro.TotalLinhas(); i++)
      {
        if (_quadro.Matriz[i, _colunaPerm].ValorSuperior < 0)
        {
          elePositivoEncontrado = true;
          break; //sair do for
        }
      }

      //se nao existir elemento negativo na mesma linha do termo livre
      //negativo, entao nao existe solucao permissivel
      if (!elePositivoEncontrado)
        return StatusSimplex.SolucaoIlimitada;

      //calcular linha permissivel
      this.CalcularLinhaPermissivel();

      return StatusSimplex.AlgoritmoTroca;

    }


    private void CalcularLinhaPermissivel()
    {
      double[] vetorQuocientes = new double[_quadro.TotalLinhas()];
      double menorQuociente = Double.MaxValue;

      //identificar a linha permissivel, verificando qual o menor quociente,
      //entre membros livres e a coluna permissivel
      for (int i = 1; i < _quadro.TotalLinhas(); i++)
      {
        //nao permitir divisao por 0 (Infinity)
        if (_quadro.Matriz[i, _colunaPerm].ValorSuperior == 0)
          vetorQuocientes[i] = -1;
        else
          vetorQuocientes[i] = _quadro.Matriz[i, 0].ValorSuperior / _quadro.Matriz[i, _colunaPerm].ValorSuperior;
      }

      //identificar qual menor valor positivo no vetor de quocientes
      for (int i = 1; i < _quadro.TotalLinhas(); i++)
        if (vetorQuocientes[i] > 0 && vetorQuocientes[i] < menorQuociente)
        {
          menorQuociente = vetorQuocientes[i];
          _linhaPerm = i;
        }
    }


    private StatusSimplex AlgoritmoTroca()
    {


      //setar o inverso do elemento permitido na sua respectiva SCS
      Celula celulaPermissivel = _quadro.Matriz[_linhaPerm, _colunaPerm];
      celulaPermissivel.ValorInferior = 1 / celulaPermissivel.ValorSuperior;



      //multiplicar toda a linha pelo EP inverso encontrado
      for (int i = 0; i < _quadro.TotalColunas(); i++)
        if (i != _colunaPerm)
          _quadro.Matriz[_linhaPerm, i].ValorInferior = _quadro.Matriz[_linhaPerm, i].ValorSuperior * celulaPermissivel.ValorInferior;


      //multiplicar toda coluna pelo -EP inverso
      for (int i = 0; i < _quadro.TotalLinhas(); i++)
        if (i != _linhaPerm)
          _quadro.Matriz[i, _colunaPerm].ValorInferior = _quadro.Matriz[i, _colunaPerm].ValorSuperior * -1 * celulaPermissivel.ValorInferior;


      //as outras posicoes da matrizInferior serao calculadas da seguinte forma:
      //Elemento = SCI da coluna permitida * SCS da linha permitida
      //correspondentes a do elemento
      for (int i = 0; i < _quadro.TotalLinhas(); i++)
        for (int j = 0; j < _quadro.TotalColunas(); j++)
          //NaN será um flag proposital setado nas sub-celulas inferiores
          if (Double.IsNaN(_quadro.Matriz[i, j].ValorInferior))
          {
            //calcular valor efetivo do elemento
            _quadro.Matriz[i, j].ValorInferior = _quadro.Matriz[i, _colunaPerm].ValorInferior * _quadro.Matriz[_linhaPerm, j].ValorSuperior;
            //realizar soma dos elementos entre SCI e SCS, gerando parte do novo quadro
            _quadro.Matriz[i, j].ValorSuperior = _quadro.Matriz[i, j].ValorInferior + _quadro.Matriz[i, j].ValorSuperior;
          }

      //transpor os elementos das SCI da linha e coluna permitidas para as suas SCS correspondentes
      for (int i = 0; i < _quadro.TotalColunas(); i++)
        _quadro.Matriz[_linhaPerm, i].ValorSuperior = _quadro.Matriz[_linhaPerm, i].ValorInferior;


      //multiplicar toda coluna pelo -EP inverso
      for (int i = 0; i < _quadro.TotalLinhas(); i++)
        _quadro.Matriz[i, _colunaPerm].ValorSuperior = _quadro.Matriz[i, _colunaPerm].ValorInferior;

      //zerar SCI, colocando NaN de forma proposital
      foreach (var celula in _quadro.Matriz)
        celula.ValorInferior = Double.NaN;

      //trocar labels da linha e coluna permissiveis
      string label = _quadro.LinhaHeader[_linhaPerm];
      _quadro.LinhaHeader[_linhaPerm] = _quadro.ColunaHeader[_colunaPerm];
      _quadro.ColunaHeader[_colunaPerm] = label;

      //mostrar quadro apos a troca
      //Console.WriteLine(_quadro.ToString());

      return StatusSimplex.PrimeiraEtapa;
    }



    public TimeSpan TempoLeituraFuncao()
    {
      return _swLeituraFuncao.Elapsed;
    }

    public TimeSpan TempoNormalizacao()
    {
      return _swNormalizacao.Elapsed;
    }

    public TimeSpan TempoOtimizacao()
    {
      return _swOtimizacao.Elapsed;
    }


    public StatusSimplex RecuperarStatus()
    {
      return this._statusSimplex;
    }


    public double RecuperarValorCusto()
    {
      return _quadro.Matriz[0, 0].ValorSuperior;
    }
  }
}
