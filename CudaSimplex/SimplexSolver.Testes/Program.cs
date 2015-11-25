using SimplexSolver.CS.Classes;
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

      Console.WriteLine("===================================");
      Console.WriteLine("=======Simplex Test Console========");
      Console.WriteLine("===================================");
      //testarSimplexSolverCS();
      testarSimplexSolverCS_MPS();
      //gerarProblemaMPS();
      Console.WriteLine();
      Console.WriteLine("Fim da execucao");
      Console.ReadKey();

    }

    private static void gerarProblemaMPS()
    {

      Console.WriteLine("Quantidade de variaveis: ");
      int qtdVariaveis = int.Parse(Console.ReadLine());

      Console.WriteLine("Quantidade de restricoes: ");
      int qtdRestricoes = int.Parse(Console.ReadLine());

      string nomeProblema = string.Concat(qtdVariaveis, "Var_", qtdRestricoes, "Rest");

      GeradorMPS gerador = new GeradorMPS(nomeProblema, qtdVariaveis, qtdRestricoes);
      string buffer = gerador.GerarProblema();

      File.WriteAllText(Path.Combine(DIRETORIO_NETLIB, nomeProblema + ".mps"), buffer);

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

      solver.Otimizar(new MPSLPReader(Path.Combine(DIRETORIO_NETLIB, "2Var_3Rest.mps"),
        new MPSLPReaderConfig
      {
        VetorRHSPossuiNome = true
      })
      , CS.Dados.Extremo.Maximizar);

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
      rest1.addVariavel("X1", 0.6);
      rest1.addVariavel("X2", 1);
      rest1.TermoLivre = 600;
      rest1.Desigualdade = CS.Dados.Desigualdade.MenorOuIgual;


      var rest2 = funcao.AddRestricao("REST_2");
      rest2.addVariavel("X1", 1);
      rest2.addVariavel("X2", 1);
      rest2.TermoLivre = 300;
      rest2.Desigualdade = CS.Dados.Desigualdade.MaiorOuIgual;

      var rest3 = funcao.AddRestricao("REST_3");
      rest3.addVariavel("X2", 1);
      rest3.TermoLivre = 100;
      rest3.Desigualdade = CS.Dados.Desigualdade.MaiorOuIgual;

      funcao.MostrarFObjetivo();

      solver.Otimizar(funcao);

    }


  }
}
