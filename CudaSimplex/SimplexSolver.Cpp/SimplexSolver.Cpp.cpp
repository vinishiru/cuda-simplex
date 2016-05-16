// SimplexSolver.Cpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int _tmain(int argc, _TCHAR* argv[])
{
  MPSReader* mpsReader = nullptr;

  FObjetivo* funcao;
  //funcao.DirecaoOtimizacao = Minimizar;
  //funcao.addVariavel("x1", 6);
  //funcao.addVariavel("x2", 12);

  //funcao.addRestricao("Rest_1");
  //funcao.addVariavelRestricao("Rest_1", "x1", 0.6);
  //funcao.addVariavelRestricao("Rest_1", "x2", 1);
  //funcao.setDesigualdadeRestricao("Rest_1", MenorOuIgual);
  //funcao.setTermoLivreRestricao("Rest_1", 600);


  cout << "========================================" << endl;
  cout << "=============Simplex Solver=============" << endl;
  cout << "========================================" << endl;
  cout << endl;

  cout << "Selecione o tipo de teste: " << endl;
  cout << endl;
  cout << "1 - DFL001 (Grande)" << endl;
  cout << "2 - KEN - 07 (Medio)" << endl;
  cout << "3 - AFIRO (Pequeno)" << endl;
  cout << "4 - Teste (Pequeno)" << endl;
  cout << "5 - Petr (Pequeno)" << endl;
  cout << "6 - ADLITTLE (Pequeno)" << endl;
  cout << "7 - SHARE2B (Pequeno)" << endl;
  cout << "8 - ISRAEL (Medio)" << endl;
  cout << "9 - CAPRI (Medio)" << endl;
  cout << "a - PILOT (Medio)" << endl;
  cout << endl;

  char x = getchar();

  cout << endl;
  cout << "Leitura da funcao objetivo iniciada..." << endl;

  string diretorio = "C:\\Users\\Shirugaron\\Source\\Repos\\cuda-simplex\\NetLib.Problemas";

  switch (x) {
  case '1':
    //Problema grande
    mpsReader = new MPSReader(diretorio + "\\DFL001.mps");
    break;

  case '2':
    //Problema medio
    mpsReader = new MPSReader(diretorio + "\\KEN-07.mps");
    break;

  case '3':
    //Problema pequeno
    mpsReader = new MPSReader(diretorio + "\\AFIRO.mps");
    break;

  case '4':
    //Problema teste
    mpsReader = new MPSReader(diretorio + "\\MPS_Test.txt");
    break;

  case '5':
    //Problema teste
    mpsReader = new MPSReader(diretorio + "\\MPS_Petr_Exemplo.txt");
    /*DESCRICAO ======================
    Neste exemplo, as condicoes de sinal das variaveis do Petr sao transformados em
    outras restricoes.
    **/

    break;

  case '6':
    mpsReader = new MPSReader(diretorio + "\\ADLITTLE.mps");
    break;

  case '7':
    mpsReader = new MPSReader(diretorio + "\\SHARE2B.mps");
    break;

  case '8':
    mpsReader = new MPSReader(diretorio + "\\ISRAEL.mps");
    break;

  case '9':
    mpsReader = new MPSReader(diretorio + "\\CAPRI.mps");
    break;

  case 'a':
    mpsReader = new MPSReader(diretorio + "\\PILOT.mps");
    break;

  case 'b':
    mpsReader = new MPSReader(diretorio + "\\2Var_3Rest.mps");
    break;

  case 'c':
    mpsReader = new MPSReader(diretorio + "\\50Var_60Rest.mps");
    break;

  case 'd':
    mpsReader = new MPSReader(diretorio + "\\500Var_500Rest.mps");
    break;

  case 'e':
    mpsReader = new MPSReader(diretorio + "\\100Var_100Rest.mps");
    break;

  case 'f':
    mpsReader = new MPSReader(diretorio + "\\1000Var_1000Rest.mps");
    break;

  case 'g':
    mpsReader = new MPSReader(diretorio + "\\250Var_250Rest.mps");
    break;

  case 'h':
    mpsReader = new MPSReader(diretorio + "\\350Var_350Rest.mps");
    break;

  case 'i':
    mpsReader = new MPSReader(diretorio + "\\750Var_750Rest.mps");
    break;

  case 'j':
    mpsReader = new MPSReader(diretorio + "\\2000Var_2000Rest.mps");
    break;

  case 'k':
    mpsReader = new MPSReader(diretorio + "\\3000Var_3000Rest.mps");
    break;

  case 'l':
    mpsReader = new MPSReader(diretorio + "\\4000Var_4000Rest.mps");
    break;

  case 'm':
    mpsReader = new MPSReader(diretorio + "\\5000Var_5000Rest.mps");
    break;


  }

  Stopwatch swLeitura;
  mpsReader->VetorRHSPossuiNome = true;
  swLeitura.Start();
  funcao = mpsReader->LerFuncaoObjetivo();
  swLeitura.Stop();



  cout << "Quantidade de variaveis: " << funcao->Variaveis.size() << endl;
  cout << "Quantidade de restricoes: " << funcao->Restricoes.size() << endl;
  cout << endl;

  cout << "Selecione a direcao da otimizacao: " << endl;
  cout << endl;
  cout << "1 - Minimizar" << endl;
  cout << "2 - Maximizar" << endl;
  cout << endl;

  getchar();
  char otimizacao = getchar();

  switch (otimizacao) {
  case '1':
    //Minimizar
    funcao->DirecaoOtimizacao = Minimizar;
    break;

  case '2':
    //Maximizar
    funcao->DirecaoOtimizacao = Maximizar;
    break;
  }

  //Normalizar funcao
  SimplexSolver solver;

  cout << "Otimizando funcao objetivo...";
  solver.otimizar(funcao);

  cout << "Otimizado!" << endl;

  cout << "Tempo leitura: " << swLeitura.Elapsed() << "s" << endl;
  cout << "Tempo normalizacao: " << solver.tempoNormalizacao() << "s" << endl;
  cout << "Tempo otimizacao: " << solver.tempoOtimizacao() << "s" << endl;
  cout << "Valor custo: " << solver.valorCusto() << endl;
  //cout << "Status final: " << solver.statusFinal() << endl;

  cout << endl << "Fim do programa. Digite qualquer tecla para sair..." << endl;
  getchar();
  getchar();

  return 0;
}

