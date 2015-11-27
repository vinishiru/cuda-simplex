using SimplexSolver.CS.Dados;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.Testes
{
  public class TranscritorMPS
  {

    const string NOME_FUNCAO_CUSTO = "COST";
    const string PREFIXO_RESTRICAO = "REST_";
    const string PREFIXO_VARIAVEL = "X_";
    const string NOME_VETOR_RHS = "RHS1";

    const string ESPACO_CURTO = "    ";
    const string ESPACO_MEDIO = "        ";
    const string ESPACO_LONGO = "            ";

    FObjetivo _funcao;
    StringBuilder _builder;

    bool _segundaColunaFlag = false;

    public TranscritorMPS(FObjetivo funcao)
    {
      _funcao = funcao;
      _builder = new StringBuilder();
    }

    public string TranscreverProblema()
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
      _builder.AppendLine(string.Format("NAME{0}{1}", ESPACO_CURTO, _funcao.NomeProblema));
    }

    private void EscreverRows()
    {
      _builder.AppendLine("ROWS");
      _builder.AppendLine(string.Format("N{0}{1}", ESPACO_CURTO, NOME_FUNCAO_CUSTO));

      foreach (var rest in _funcao.Restricoes.Values)
        _builder.AppendLine(string.Concat(ValorDesigualdadeToString((int)rest.Desigualdade), ESPACO_CURTO, rest.Nome));

    }

    private void EscreverColumns()
    {

      Variavel varRest = null;

      _builder.AppendLine("COLUMNS");

      foreach (var variavel in _funcao.Variaveis.Values)
      {

        //sempre escrever valor da funcao de custo
        EscreverValorColumn(NOME_FUNCAO_CUSTO, variavel);

        foreach (var rest in _funcao.Restricoes.Values)
        {
          varRest = rest.Variaveis.Values.FirstOrDefault(m => m.Nome.Equals(variavel.Nome));
          if (varRest != null)
            EscreverValorColumn(rest.Nome, varRest);
        }
        //finaliza a linha caso esteja no meio da segunda coluna
        if (_segundaColunaFlag)
        {
          _builder.AppendLine();
          _segundaColunaFlag = false;
        }

      }
    }

    private void EscreverValorColumn(string nomeFuncao, Variavel variavel)
    {

      if (!_segundaColunaFlag)
        _builder.Append(string.Concat(ESPACO_CURTO, variavel.Nome, ESPACO_CURTO, nomeFuncao, ESPACO_LONGO, variavel.Coeficiente));
      else
        _builder.AppendLine(string.Concat(ESPACO_CURTO, nomeFuncao, ESPACO_LONGO, variavel.Coeficiente));

      _segundaColunaFlag = !_segundaColunaFlag;
    }

    private void EscreverRHS()
    {

      _builder.AppendLine("RHS");

      foreach (var rest in _funcao.Restricoes.Values)
        EscreverValorRHS(rest);

      if (_segundaColunaFlag)
      {
        _builder.AppendLine();
        _segundaColunaFlag = false;
      }
    }

    private void EscreverValorRHS(Restricao rest)
    {

      if (!_segundaColunaFlag)
        _builder.Append(string.Concat(ESPACO_CURTO, NOME_VETOR_RHS, ESPACO_CURTO, rest.Nome, ESPACO_LONGO, rest.TermoLivre));
      else
        _builder.AppendLine(string.Concat(ESPACO_CURTO, rest.Nome, ESPACO_LONGO, rest.TermoLivre));

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
