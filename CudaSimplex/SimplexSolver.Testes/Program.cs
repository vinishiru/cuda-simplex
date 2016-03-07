using SimplexSolver.CS.Classes;
using SimplexSolver.CS.Dados;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplexSolver.Testes
{
  class Program
  {

    private const string DIRETORIO_NETLIB = "..\\..\\..\\..\\NetLib.Problemas";

    static void Main(string[] args)
    {
      string key = string.Empty;
      do
      {
        Console.Clear();
        Console.WriteLine("===================================");
        Console.WriteLine("=======Simplex Test Console========");
        Console.WriteLine("===================================");

        Console.WriteLine();

        Console.WriteLine("1 - Gerar Problema");
        Console.WriteLine("2 - Resolver Problema");
        Console.WriteLine("3 - Formatar Problemas");
        Console.WriteLine();

        Console.WriteLine("Opcao:");
        key = Console.ReadLine();

        switch (key)
        {
          case "1":
            gerarProblemaLP();
            break;
          case "2":
            testarSimplexSolverCS_MPS();
            break;
          case "3":
            formatarProblemas();
            break;
        }

        Console.WriteLine();
        Console.WriteLine("Fim da execucao");

        Console.WriteLine("Sair? s/n");
        Console.ReadKey();

      } while (key != "s");
    }


    private static void formatarProblemas()
    {
      FormatadorMPS formatMPS = null;

      foreach (var file in Directory.GetFiles(DIRETORIO_NETLIB, "*Var*Rest.mps"))
      {
        Console.WriteLine(string.Format("Formatando arquivo {0} ...", file));
        formatMPS = new FormatadorMPS(file);
        formatMPS.FormatarArquivo();
        Console.WriteLine(string.Format("Arquivo formatado! {0}", file));
        Console.WriteLine();
      }

    }

    private static void gerarProblemaLP()
    {

      Console.WriteLine("Quantidade de variaveis: ");
      int qtdVariaveis = int.Parse(Console.ReadLine());

      Console.WriteLine("Quantidade de restricoes: ");
      int qtdRestricoes = int.Parse(Console.ReadLine());

      Console.WriteLine("Densidade: ");
      int densidade = int.Parse(Console.ReadLine());


      GeradorLP gerador = new GeradorLP(qtdVariaveis, qtdRestricoes, densidade);
      FObjetivo funcao = gerador.GerarProblemaLP();

      Console.WriteLine();
      //funcao.MostrarFObjetivo();

      TranscritorMPS transcritor = new TranscritorMPS(funcao);
      string buffer = transcritor.TranscreverProblema();

      File.WriteAllText(Path.Combine(DIRETORIO_NETLIB, funcao.Nome + ".mps"), buffer);

    }

    private static void testarSimplexSolverCS_MPS()
    {
      string[] problemas = {
                             "MPS_Petr_Exemplo.txt",
                             "MPS_Test.txt",
                             "AFIRO.mps",
                             "SHARE2B.mps",
                             "ADLITTLE.mps",
                             "CAPRI.mps",
                             "ISRAEL.mps",
                             "KEN-07.mps",
                             "DFL001.mps"
                           };


      SimplexSolver.CS.SimplexSolverCPU solver = new CS.SimplexSolverCPU();

      solver.Otimizar(new MPSLPReader(Path.Combine(DIRETORIO_NETLIB, "KEN-07.mps"),
        new MPSLPReaderConfig
      {
        VetorRHSPossuiNome = true
      })
      , CS.Dados.Extremo.Minimizar);

      Console.WriteLine();
      Console.WriteLine("Tempo leitura: {0}", solver.TempoLeituraFuncao().TotalSeconds);
      Console.WriteLine("Tempo normalizacao: {0}", solver.TempoNormalizacao().TotalSeconds);
      Console.WriteLine("Tempo otimizacao: {0}", solver.TempoOtimizacao().TotalSeconds);
      Console.WriteLine("Valor custo: {0}", solver.RecuperarValorCusto());
      Console.WriteLine("Status final: {0}", solver.RecuperarStatus());
      Console.WriteLine();

    }

    private static void testarSimplexSolverCS()
    {

      SimplexSolver.CS.SimplexSolverCPU solver = new CS.SimplexSolverCPU();
      SimplexSolver.CS.Dados.FObjetivo funcao = new CS.Dados.FObjetivo();

      funcao.AddVariavel("X1", 6);
      funcao.AddVariavel("X2", 12);

      var rest1 = funcao.AddRestricao("REST_1");
      rest1.AddVariavel("X1", 0.6);
      rest1.AddVariavel("X2", 1);
      rest1.TermoLivre = 600;
      rest1.Desigualdade = CS.Dados.Desigualdade.MenorOuIgual;


      var rest2 = funcao.AddRestricao("REST_2");
      rest2.AddVariavel("X1", 1);
      rest2.AddVariavel("X2", 1);
      rest2.TermoLivre = 300;
      rest2.Desigualdade = CS.Dados.Desigualdade.MaiorOuIgual;

      var rest3 = funcao.AddRestricao("REST_3");
      rest3.AddVariavel("X2", 1);
      rest3.TermoLivre = 100;
      rest3.Desigualdade = CS.Dados.Desigualdade.MaiorOuIgual;

      funcao.MostrarFObjetivo();

      solver.Otimizar(funcao);

    }


  }
}
