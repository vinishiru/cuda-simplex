#include "SimplexSolver.h"

void SimplexSolver::otimizar(ILPReader* reader){
  this->otimizar(reader->LerFuncaoObjetivo());
}

void SimplexSolver::otimizar(FObjetivo* func){

  //instanciar quadro com funcao objetivo
  this->quadro = new Quadro(func);

  //atribuir valores iniciais para controle de linha e coluna permissiveis
  this->linhaPerm = -1;
  this->colunaPerm = -1;
  //limpar historico
  this->historico.clear();
  //definir status inicial do algoritmo
  this->status = PrimeiraEtapa;

  this->swNormalizacao.Start();
  this->quadro->buildQuadro();
  this->swNormalizacao.Stop();

  int qtdIteracoes = 1;

  cout << endl;

  //mostrar quadro atualizado
  //this->quadro->toString();

  Stopwatch swPrimeiraEtapa;
  Stopwatch swSegundaEtapa;
  Stopwatch swAlgTroca;

  double tempoTotalPrimeiraEtapa = 0, tempoTotalSegundaEtapa = 0, tempoTotalTroca = 0;

  int auxLinhaPerm = 0, auxColunaPerm = 0;


  try
  {
    this->swOtimizacao.Start();
    this->swSegPorIteracao.Start();

    while (this->status != SolucaoOtima &&
      this->status != SolucaoIlimitada &&
      this->status != SolucaoImpossivel)
    {

      this->historico.push_back(this->status);

      switch (this->status)
      {
      case PrimeiraEtapa:
        swPrimeiraEtapa.Start();
        this->status = this->algoritmoPrimeiraEtapa();
        swPrimeiraEtapa.Stop();
        break;

      case SegundaEtapa:
        swSegundaEtapa.Start();
        this->status = this->algoritmoSegundaEtapa();
        swSegundaEtapa.Stop();
        break;

      case AlgoritmoTroca:

        //gravar linha e coluna permitidas encontradas
        auxLinhaPerm = this->linhaPerm;
        auxColunaPerm = this->colunaPerm;

        swAlgTroca.Start();
        this->status = this->algoritmoTroca();
        swAlgTroca.Stop();
        this->swSegPorIteracao.Stop();

        tempoTotalPrimeiraEtapa += swPrimeiraEtapa.Elapsed();
        tempoTotalSegundaEtapa += swSegundaEtapa.Elapsed();
        tempoTotalTroca += swAlgTroca.Elapsed();

        if (qtdIteracoes % 1000 == 0){
          //logar tempos
          cout << "==================================" << endl;
          //logar linha e coluna permitidas encontradas
          cout << "Linha:\t" << this->linhaPerm;
          cout << "\tColuna:\t" << this->colunaPerm;
          cout << "\tIteracao\t" << qtdIteracoes << endl;
          cout << "Primeira etapa:\t\t" << swPrimeiraEtapa.Elapsed() << endl;
          cout << "Segunda etapa:\t\t" << swSegundaEtapa.Elapsed() << endl;
          cout << "Algoritmo Troca:\t" << swAlgTroca.Elapsed() << endl;
          cout << "Total:\t\t\t" << this->swSegPorIteracao.Elapsed() << endl;
          cout << "==================================" << endl;
        }

        //reiniciar tempo de iteracao
        this->swSegPorIteracao.Start();
        qtdIteracoes++;

        break;
      }

      //this->quadro->toString();

    }

    //parar timer
    this->swOtimizacao.Stop();

    //guardar ultimo status
    this->historico.push_back(this->status);

    //logada final
    //logar tempos
    cout << "==================================" << endl;
    //logar linha e coluna permitidas encontradas
    cout << "Iteracao\t" << qtdIteracoes << endl;
    cout << "Media Primeira etapa:\t" << tempoTotalPrimeiraEtapa / qtdIteracoes << endl;
    cout << "Media Segunda etapa:\t" << tempoTotalSegundaEtapa / qtdIteracoes << endl;
    cout << "Media Algoritmo Troca:\t" << tempoTotalTroca / qtdIteracoes << endl;
    cout << "Media Seg/Iteracao:\t" << this->swOtimizacao.Elapsed() / qtdIteracoes << endl;
    cout << "==================================" << endl;

  }
  catch (exception e)
  {
    cout << endl << "Ocorreu um erro no algoritmo!" << e.what() << endl;
  }

}

StatusSimplex SimplexSolver::algoritmoPrimeiraEtapa(){

  bool tlNegativoEncontrado = false;
  bool elemNegativoEncontrado = false;
  int linhaTLNegativo = 0;

  //zerar linha e coluna permissiveis
  this->colunaPerm = -1;
  this->linhaPerm = -1;

  //procurar por um elemento negativo na coluna dos termos livres,
  //escolhendo o primeiro elemento negativo encontrado
  for (int i = 0; i < this->quadro->totalLinhas; i++){
    if (this->quadro->matrizSuperior[i * this->quadro->totalColunas] < 0){
      linhaTLNegativo = i;
      tlNegativoEncontrado = true;
      break; //sair do for
    }
  }

  //Se o elemento negativo nao existe, ir para segunda etapa do
  //algoritmo
  if (!tlNegativoEncontrado)
    return SegundaEtapa;

  //procurar por elemento negativo na linha do termo livre negativo,
  //escolhendo o primeiro elemento negativo encontrado
  for (int j = 1; j < this->quadro->totalColunas; j++){
    if (this->quadro->matrizSuperior[linhaTLNegativo * this->quadro->totalColunas + j] < 0){
      this->colunaPerm = j;
      elemNegativoEncontrado = true;
      break; //sair do for
    }
  }

  //se nao existir elemento negativo na mesma linha do termo livre
  //negativo, entao nao existe solucao permissivel
  if (!elemNegativoEncontrado)
    return SolucaoImpossivel;

  //calcular linha permissivel
  this->calcularLinhaPermissivel();

  return AlgoritmoTroca;
}

StatusSimplex SimplexSolver::algoritmoSegundaEtapa(){

  bool colunaFPositivoEncontrado = false;
  bool elePositivoEncontrado = false;

  //zerar linha e coluna permissiveis
  this->colunaPerm = -1;
  this->linhaPerm = -1;

  //na linha F(X) procuramos por um elemento positivo,
  //desconsiderando o termo livre
  for (int i = 1; i < this->quadro->totalColunas; i++){
    if (this->quadro->matrizSuperior[i] > 0){
      this->colunaPerm = i;
      colunaFPositivoEncontrado = true;
      break;
    }
  }

  //se nao encontrarmos elemento positivo na linha F(X),
  //significa que a solucao atual e otima
  if (!colunaFPositivoEncontrado)
    return SolucaoOtima;

  //usando a coluna F(x) de elemento positivo,
  //tentar encontrar um elemento positivo nas linhas abaixo,
  //da linha F(x), serve apenas para verificar se a solucao nao e ilimitada
  for (int i = 1; i < this->quadro->totalLinhas; i++){
    if (this->quadro->matrizSuperior[i * this->quadro->totalColunas + this->colunaPerm] < 0){
      elePositivoEncontrado = true;
      break; //sair do for
    }
  }

  //se nao existir elemento negativo na mesma linha do termo livre
  //negativo, entao nao existe solucao permissivel
  if (!elePositivoEncontrado)
    return SolucaoIlimitada;

  //calcular linha permissivel
  this->calcularLinhaPermissivel();

  return AlgoritmoTroca;
}

void SimplexSolver::calcularLinhaPermissivel(){

  float *vetorQuocientes;
  float menorQuociente = FLT_MAX;

  //identificar a linha permissivel, verificando qual o menor quociente,
  //entre membros livres e a coluna permissivel
  vetorQuocientes = simplexGPU.calcularQuocientes(this->quadro, this->colunaPerm);

  //identificar qual menor valor positivo no vetor de quocientes
  for (int i = 1; i < this->quadro->totalLinhas; i++){
    if (vetorQuocientes[i] > 0 && vetorQuocientes[i] < menorQuociente){
      menorQuociente = vetorQuocientes[i];
      this->linhaPerm = i;
    }
  }
}

StatusSimplex SimplexSolver::algoritmoTroca(){

  //executarl algoritmo de GPU para atualizar o quadro Simplex
  simplexGPU.atualizarQuadro(this->quadro, this->colunaPerm, this->linhaPerm);

  //trocar labels da linha e coluna permissiveis
  string rowLabel = this->quadro->rowHeader.at(this->linhaPerm);
  this->quadro->rowHeader.at(this->linhaPerm) = this->quadro->colHeader.at(this->colunaPerm);
  this->quadro->colHeader.at(this->colunaPerm) = rowLabel;


  return PrimeiraEtapa;
}

double SimplexSolver::tempoLeitura(){
  return 0.0;
}

double SimplexSolver::tempoNormalizacao(){
  return this->swNormalizacao.Elapsed();
}

double SimplexSolver::tempoOtimizacao(){
  return this->swOtimizacao.Elapsed();
}

float SimplexSolver::valorCusto(){
  return this->quadro->matrizSuperior[0];
}

StatusSimplex SimplexSolver::statusFinal(){
  return this->status;
}