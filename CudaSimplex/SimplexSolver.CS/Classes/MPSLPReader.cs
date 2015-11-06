using SimplexSolver.CS.Dados;
using SimplexSolver.CS.Interfaces;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.CS.Classes
{
  public class MPSLPReader : ILPReader
  {

    private CultureInfo _culture = new CultureInfo("en");

    private StreamReader _mpsFile;
    private FObjetivo _funcao;
    private string _path;
    private string _linha;

    private string[] _tokens;
    private char[] _separadores = new char[] { ' ' };

    public MPSLPReader(string path)
    {

      if (string.IsNullOrEmpty(path))
        throw new ArgumentNullException("path");

      if (!File.Exists(path))
        throw new IOException("Arquivo nao existe.");

      _path = path;
      _mpsFile = new StreamReader(path, Encoding.UTF8);
      _funcao = new FObjetivo();

    }

    ~MPSLPReader()
    {
      _mpsFile.Close();
    }

    public Dados.FObjetivo LerFuncaoObjetivo()
    {

      readNAME();
      readROWS();
      readCOLUMNS();
      readRHS();
      //Temos que verificar se realmente sera necessario utilizar as restricoes
      //finais, chamadas BOUNDS do arquivo MPS.
      //readBOUNDS();

      _mpsFile.Close();

      return this._funcao;
    }

    private void readBOUNDS()
    {
      /*				type            meaning
                  ---------------------------------------------------
                    LO    lower bound        b <= x (< +inf)
                    UP    upper bound        (0 <=) x <= b
                    FX    fixed variable     x = b
                    FR    free variable      -inf < x < +inf
                    MI    lower bound -inf   -inf < x (<= 0)
                    PL    upper bound +inf   (0 <=) x < +inf
                    BV    binary variable    x = 0 or 1
                    LI    integer variable   b <= x (< +inf)
                    UI    integer variable   (0 <=) x <= b
                    SC    semi-cont variable x = 0 or l <= x <= b
                          l is the lower bound on the variable
                          If none set then defaults to 1*/

      //Como se trata de um bloco opcional dos problemas,
      //Apenas tratarei os tipos LO e UP.
      Restricao auxRest;
      string boundType;
      string boundName;
      string nomeVariavel;
      string boundValue;
      string auxRestName; //sera criada uma nova restricao se a fronteira for valida
      Desigualdade tipoDesigualdade = Desigualdade.Igual;

      if (_linha.Equals("BOUNDS"))
      {
        _linha = _mpsFile.ReadLine();

        while (!_linha.Equals("ENDATA"))
        {

          _tokens = GerarTokens(_linha);

          boundType = _tokens[0];
          boundName = _tokens[1];
          nomeVariavel = _tokens[2];
          boundValue = _tokens[3];

          if (boundType.Equals("LO"))
            tipoDesigualdade = Desigualdade.MaiorOuIgual;
          else if (boundType.Equals("UP"))
            tipoDesigualdade = Desigualdade.MenorOuIgual;

          auxRest = _funcao.AddRestricao();
          //Configurar nova restricao
          _funcao.AddVariavelRestricao(auxRest.Nome, nomeVariavel, 1.0);
          _funcao.SetDesigualdadeRestricao(auxRest.Nome, tipoDesigualdade);
          _funcao.SetTermoLivreRestricao(auxRest.Nome, double.Parse(boundValue, _culture));

          _linha = _mpsFile.ReadLine();

        }
      }

    }

    private void readRHS()
    {
      string nomeVetorRHS;
      string nomeFuncao;
      string valorTermoLivre;

      if (_linha.Equals("RHS"))
      {
        _linha = _mpsFile.ReadLine();

        while (!string.IsNullOrEmpty(_linha) &&
          !_linha.Equals("BOUNDS") &&
          !_linha.Equals("ENDATA"))
        {
          _tokens = GerarTokens(_linha);

          nomeVetorRHS = _tokens[0];
          nomeFuncao = _tokens[1];
          valorTermoLivre = _tokens[2];

          if (nomeFuncao.Equals(_funcao.Nome))
            _funcao.TermoLivre = double.Parse(valorTermoLivre, _culture);
          else
            _funcao.SetTermoLivreRestricao(nomeFuncao, double.Parse(valorTermoLivre, _culture));

          //Se tiver mais tokens, ler mais duas colunas
          //Nome Variavel e Valor Variavel, obedecendo as mesmas regras anteriores
          if (_tokens.Length > 3)
          {
            nomeFuncao = _tokens[3];
            valorTermoLivre = _tokens[4];

            if (nomeFuncao.Equals(_funcao.Nome))
              _funcao.TermoLivre = double.Parse(valorTermoLivre, _culture);
            else
              _funcao.SetTermoLivreRestricao(nomeFuncao, double.Parse(valorTermoLivre, _culture));

          }

          _linha = _mpsFile.ReadLine();

        }

      }
    }

    private void readCOLUMNS()
    {
      string nomeFuncao;
      string nomeVariavel;
      string valorVariavel;

      if (_linha.Equals("COLUMNS"))
      {
        _linha = _mpsFile.ReadLine();

        while (!_linha.Equals("RHS"))
        {
          _tokens = GerarTokens(_linha);

          //Ler no minimo 3 tokens
          //Nome variavel / Nome Funcao|Restricao / Valor variavel

          nomeVariavel = _tokens[0];
          nomeFuncao = _tokens[1];
          valorVariavel = _tokens[2];

          //Se nomeFuncao for o nome da funcao objetivo, atribuir variaveis nela
          if (nomeFuncao.Equals(_funcao.Nome))
            _funcao.AddVariavel(nomeVariavel, double.Parse(valorVariavel, _culture));

          //Se nao, atribuir variaveis a restricao
          else
            _funcao.AddVariavelRestricao(nomeFuncao, nomeVariavel, double.Parse(valorVariavel, _culture));

          //Se tiver mais tokens, ler mais duas colunas
          //Nome Variavel e Valor Variavel, obedecendo as mesmas regras anteriores
          if (_tokens.Length > 3)
          {
            nomeFuncao = _tokens[3];
            valorVariavel = _tokens[4];

            //Se nomeFuncao for o nome da funcao objetivo, atribuir variaveis nela
            if (nomeFuncao.Equals(_funcao.Nome))
              _funcao.AddVariavel(nomeVariavel, double.Parse(valorVariavel, _culture));

            //Se nao, atribuir variaveis a restricao
            else
              _funcao.AddVariavelRestricao(nomeFuncao, nomeVariavel, double.Parse(valorVariavel, _culture));

          }

          _linha = _mpsFile.ReadLine();
        }

      }
    }

    private void readROWS()
    {
      Restricao rest;
      string desigualdade;
      string nomeFuncao;

      _linha = _mpsFile.ReadLine();
      _tokens = GerarTokens(_linha);

      if (_tokens[0].Equals("ROWS"))
      {
        _linha = _mpsFile.ReadLine();

        while (!_linha.Equals("COLUMNS"))
        {
          _tokens = GerarTokens(_linha);

          desigualdade = _tokens[0];
          nomeFuncao = _tokens[1];

          switch (desigualdade)
          {
            case "N":
              _funcao.Nome = nomeFuncao;
              break;

            case "E":
              rest = _funcao.AddRestricao(nomeFuncao);
              rest.Desigualdade = Desigualdade.Igual;
              break;

            case "L":
              rest = _funcao.AddRestricao(nomeFuncao);
              rest.Desigualdade = Desigualdade.MenorOuIgual;
              break;

            case "G":
              rest = _funcao.AddRestricao(nomeFuncao);
              rest.Desigualdade = Desigualdade.MaiorOuIgual;
              break;

            default:
              throw new ApplicationException("Sintaxe MPS invalida.");
          }

          //ler nova linha
          _linha = _mpsFile.ReadLine();
        }
      }


    }

    private void readNAME()
    {
      _linha = _mpsFile.ReadLine();

      string[] nomeProblema = _linha.Split(_separadores, StringSplitOptions.RemoveEmptyEntries);

      _funcao.NomeProblema = nomeProblema[1];

    }

    private string[] GerarTokens(string linha)
    {
      return linha.Split(_separadores, StringSplitOptions.RemoveEmptyEntries);
    }
  }
}
