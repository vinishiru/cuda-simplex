using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.Testes
{
  public class FormatadorMPS
  {


    /*
     * The following template is a guide for the use of MPS format:

---------------------------------------------------------------------
Field:    1           2          3         4         5         6
Columns:  2-3        5-12      15-22     25-36     40-47     50-61

          NAME   problem name

          ROWS

           type     name

          COLUMNS
                   column       row       value     row      value
                    name        name                name
          RHS
                    rhs         row       value     row      value
                    name        name                name
          RANGES
                    range       row       value     row      value
                    name        name                name
          BOUNDS

           type     bound       column    value
                    name        name

          SOS
           type     CaseName    SOSName   SOSpriority
                    CaseName    VarName1  VarWeight1
                    CaseName    VarName2  VarWeight2

                    CaseName    VarNameN  VarWeightN

          ENDATA
     * */

    private string[] KEY_WORDS = { "NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "SOS", "ENDATA" };
    private string[] ESPACO = new string[] { " " };

    private FileInfo _file;
    private StringBuilder _outBuffer;
    private string[] _lines;
    private int _rowIndex = 0;
    private StreamReader _stream = null;
    private string[] tokens = null;

    public class Coluna
    {
      public int PosInicial { get; set; }
      public int Tamanho { get; set; }
      public int Salto { get; set; }
      public bool Inverso { get; set; }
      public int Total
      {
        get
        {
          return Salto + Tamanho;
        }
      }
    }

    public static Coluna PrimeiraCol = new Coluna { PosInicial = 2, Tamanho = 2, Salto = 1 };
    public static Coluna SegundaCol = new Coluna { PosInicial = 5, Tamanho = 8, Salto = 1 };
    public static Coluna TerceiraCol = new Coluna { PosInicial = 15, Tamanho = 8, Salto = 2 };
    public static Coluna QuartaCol = new Coluna { PosInicial = 25, Tamanho = 12, Salto = 2, Inverso = true };
    public static Coluna QuintaCol = new Coluna { PosInicial = 40, Tamanho = 8, Salto = 3 };
    public static Coluna SextaCol = new Coluna { PosInicial = 50, Tamanho = 12, Salto = 2, Inverso = true };


    public FormatadorMPS()
    {
      _outBuffer = new StringBuilder();
    }

    public FormatadorMPS(string fileName)
      : this()
    {

      if (!File.Exists(fileName))
        throw new FileNotFoundException();

      if (!Path.GetExtension(fileName).Equals(".mps"))
        throw new InvalidOperationException("Arquivo não é MPS.");

      this._file = new FileInfo(fileName);

    }

    public void FormatarArquivo()
    {

      _stream = new StreamReader(_file.FullName);

      FormatarCabecalho();
      FormatarRows();
      FormatarColumns();
      FormatarRHS();

      FinalizarFormatacao();

    }

    private void FinalizarFormatacao()
    {

      //fechar stream de leitura
      _stream.Close();

      //sobrescresver arquivo
      File.WriteAllText(Path.Combine(_file.DirectoryName, Path.GetFileNameWithoutExtension(_file.Name) + "_F.mps"), _outBuffer.ToString());
    }

    private void FormatarCabecalho()
    {

      string line = _stream.ReadLine();

      while (!line.Contains("ROWS"))
      {
        _outBuffer.AppendLine(line);
        line = _stream.ReadLine();
      }
      _outBuffer.AppendLine("ROWS");
    }

    private void FormatarRows()
    {
      //le rows
      string line = _stream.ReadLine();

      while (!line.Contains("COLUMNS"))
      {
        EscreverPadraoDuasColunas(line);
        line = _stream.ReadLine();
        _outBuffer.AppendLine();
      }

      _outBuffer.AppendLine("COLUMNS");
    }

    private void FormatarColumns()
    {
      //le rows
      string line = _stream.ReadLine();

      while (!line.Contains("RHS"))
      {
        EscreverPadraoSeisColunas(line);
        line = _stream.ReadLine();
        _outBuffer.AppendLine();
      }

      _outBuffer.AppendLine("RHS");
    }

    private void FormatarRHS()
    {
      //le rows
      string line = _stream.ReadLine();

      while (!line.Contains("ENDATA"))
      {
        EscreverPadraoSeisColunas(line);
        line = _stream.ReadLine();
        _outBuffer.AppendLine();
      }

      _outBuffer.AppendLine("ENDATA");
    }

    private void EscreverPadraoDuasColunas(string line)
    {
      tokens = line.Split(ESPACO, StringSplitOptions.RemoveEmptyEntries);

      _outBuffer.Append(FormatarColuna(tokens[0], PrimeiraCol));
      _outBuffer.Append(FormatarColuna(tokens[1], SegundaCol));
    }

    private void EscreverPadraoSeisColunas(string line)
    {
      tokens = line.Split(ESPACO, StringSplitOptions.RemoveEmptyEntries);

      _outBuffer.Append(FormatarColuna(string.Empty, PrimeiraCol));

      _outBuffer.Append(FormatarColuna(tokens[0], SegundaCol));
      _outBuffer.Append(FormatarColuna(tokens[1], TerceiraCol));
      _outBuffer.Append(FormatarColuna(tokens[2], QuartaCol));

      if (tokens.Length > 3)
      {
        _outBuffer.Append(FormatarColuna(tokens[3], QuintaCol));
        _outBuffer.Append(FormatarColuna(tokens[4], SextaCol));
      }
    }

    public string FormatarColuna(string token, Coluna coluna)
    {
      //sempre pular primeira coluna
      StringBuilder builder = new StringBuilder();

      //verificar truncar
      token = TratarToken(token);

      //concatenar salto
      builder.Append(' ', coluna.Salto);

      if (!coluna.Inverso)
      {
        builder.Append(token);
        builder.Append(' ', coluna.Tamanho - token.Length);
      }
      else
      {
        builder.Append(' ', coluna.Tamanho - token.Length);
        builder.Append(token);
      }

      return builder.ToString();

    }

    private string TratarToken(string token)
    {
      //tratar nome da restricao
      if (token.Contains("REST_"))
      {
        var restSplit = token.Split(new string[] { "_" }, StringSplitOptions.RemoveEmptyEntries);
        token = string.Concat("R", restSplit[1]);
      }
      else if (token.Contains("X_"))
      {
        var restSplit = token.Split(new string[] { "_" }, StringSplitOptions.RemoveEmptyEntries);
        token = string.Concat("X", restSplit[1]);
      }

      return token;
    }

  }
}
