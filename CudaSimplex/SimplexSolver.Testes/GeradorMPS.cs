using SimplexSolver.CS.Dados;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.Testes
{
  public class GeradorMPS
  {

    private string _nome;

    public const string NOME_FUNCAO_CUSTO = "COST";
    public const string PREFIXO_RESTRICAO = "R";
    public const string PREFIXO_VARIAVEL = "X";
    public const string NOME_VETOR_RHS = "RHS1";

    private const string ESPACO_CURTO = "    ";
    private const string ESPACO_MEDIO = "        ";
    private const string ESPACO_LONGO = "            ";

    private int _qtdVariaveis;
    private int _qtdRestricoes;
    private StringBuilder _builder;
    private List<String> _nomesRestricoes;
    private Random _rnd;

    private bool _segundaColunaFlag = false;

    public GeradorMPS(string nome, int qtdVariaveis, int qtdRestricoes)
    {
      this._nome = nome;
      this._qtdVariaveis = qtdVariaveis;
      this._qtdRestricoes = qtdRestricoes;
      this._rnd = new Random();

      //cria a lista de restricoes e ja adiciona a primeira funcao de custo
      _nomesRestricoes = new List<string>();

      _builder = new StringBuilder();
    }

    public string GerarProblema()
    {
      EscreverCabecalho();
      EscreverRows();
      EscreverColumns();
      EscreverRHS();

      _builder.AppendLine("ENDATA");

      return this._builder.ToString();
    }

    private void EscreverCabecalho()
    {
      _builder.AppendLine(string.Format("NAME{0}{1}", ESPACO_CURTO, _nome));
    }

    private void EscreverRows()
    {
      _builder.AppendLine("ROWS");
      _builder.AppendLine(string.Format("N{0}{1}", ESPACO_CURTO, NOME_FUNCAO_CUSTO));

      int desigualdade = 0;
      string nomeRest = string.Empty;

      for (int i = 0; i < _qtdRestricoes; i++)
      {
        desigualdade = _rnd.Next(0, 3);
        nomeRest = PREFIXO_RESTRICAO + i;
        _nomesRestricoes.Add(nomeRest);
        _builder.AppendLine(string.Concat(ValorDesigualdadeToString(desigualdade), ESPACO_CURTO, nomeRest));
      }
    }

    private void EscreverColumns()
    {

      string nomeVar = string.Empty;

      _builder.AppendLine("COLUMNS");

      for (int i = 0; i < _qtdVariaveis; i++)
      {
        nomeVar = PREFIXO_VARIAVEL + i;

        //sempre escrever valor da funcao de custo
        EscreverValorColumn(NOME_FUNCAO_CUSTO, nomeVar);

        foreach (string rest in _nomesRestricoes)
          //probabilidade para gerar problema semi-esparso
          if (_rnd.Next(0, 3) < 1)
            EscreverValorColumn(rest, nomeVar);

        //finaliza a linha caso esteja no meio da segunda coluna
        if (_segundaColunaFlag)
        {
          _builder.AppendLine();
          _segundaColunaFlag = false;
        }

      }
    }

    private void EscreverValorColumn(string nomeFuncao, string nomeVar)
    {

      //gerar valor
      int valor = _rnd.Next(-2, 9);

      if (!_segundaColunaFlag)
        _builder.Append(string.Concat(ESPACO_CURTO, nomeVar, ESPACO_CURTO, nomeFuncao, ESPACO_LONGO, valor));
      else
        _builder.AppendLine(string.Concat(ESPACO_CURTO, nomeFuncao, ESPACO_LONGO, valor));

      _segundaColunaFlag = !_segundaColunaFlag;
    }

    private void EscreverRHS()
    {

      _builder.AppendLine("RHS");

      foreach (var nomeRest in _nomesRestricoes)
        EscreverValorRHS(nomeRest);

      if (_segundaColunaFlag)
      {
        _builder.AppendLine();
        _segundaColunaFlag = false;
      }
    }

    private void EscreverValorRHS(string nomeFuncao)
    {

      //gerar valor
      int valor = _rnd.Next(-1, 9);

      if (!_segundaColunaFlag)
        _builder.Append(string.Concat(ESPACO_CURTO, NOME_VETOR_RHS, ESPACO_CURTO, nomeFuncao, ESPACO_LONGO, valor));
      else
        _builder.AppendLine(string.Concat(ESPACO_CURTO, nomeFuncao, ESPACO_LONGO, valor));

      _segundaColunaFlag = !_segundaColunaFlag;
    }


    private string ValorDesigualdadeToString(int desigualdade)
    {
      Desigualdade enumDes = (Desigualdade)Enum.Parse(typeof(Desigualdade), desigualdade.ToString());

      if (enumDes == Desigualdade.Maior || enumDes == Desigualdade.MaiorOuIgual)
        return "G";

      if (enumDes == Desigualdade.Menor || enumDes == Desigualdade.MenorOuIgual)
        return "L";

      return "L";
    }
  }
}
