#include "stdafx.h"


void SimplexSolver::otimizar(FObjetivo *func){

  this->quadro = new Quadro(func);

  //atribuir valores iniciais para controle de linha e coluna permissiveis
  this->linhaPerm = -1;
  this->colunaPerm = -1;

  //definir status inicial do algoritmo
  this->status = PrimeiraEtapa;

  this->swNormalizacao.Start();
  this->quadro->buildQuadro();
  this->swNormalizacao.Stop();

  //exibir quadro montado;
  //this->quadro->toString();

  //Desenvolvimento do Algoritmo Tradicional - Abordagem Surplus Variable
  //Inicialmente, deve-se verificar se o problema possui variáveis básicas
  //com coeficiente negativo. Isso significa que o problema nao esta na forma padrao
  //que difine que todas as variaveis devem ser nao-negativas.

  //Pivotamento - Algoritmo de troca
  //Transformar o elemento permissivo em 1, usando o valor auxiliar por toda linha
  //permissiva.
  //Transformar os demais elementos da coluna permissiva em 0, operando por toda linha
  //correspondente.


  int qtdIteracoes = 1;

  try
  {
    this->swOtimizacao.Start();
    this->swSegPorIteracao.Start();

    while (this->status != SolucaoOtima &&
      this->status != SolucaoIlimitada &&
      this->status != SolucaoImpossivel)
    {

      //this->historico.push_back(this->status);

      switch (this->status)
      {
      case PrimeiraEtapa:
        this->status = this->algoritmoPrimeiraEtapa();
        break;

      case SegundaEtapa:
        this->status = this->algoritmoSegundaEtapa();
        break;

      case AlgoritmoTroca:

        //this->quadro->toString();
        this->status = this->algoritmoTroca();

        this->swSegPorIteracao.Stop();

        if (qtdIteracoes % 100 == 0){
          cout << "Linha:\t" << this->linhaPerm;
          cout << "\tColuna:\t" << this->colunaPerm;
          cout << "\tIteracao\t" << qtdIteracoes;
          cout << "\tTempo:\t" << this->swSegPorIteracao.Elapsed() << endl;
        }


        this->swSegPorIteracao.Start();
        qtdIteracoes++;

        break;
      }
    }

    //parar timer
    this->swOtimizacao.Stop();

    //guardar ultimo status
    //this->historico.push_back(this->status);

  }
  catch (exception e)
  {
    cout << endl << "Ocorreu um erro no algoritmo!" << e.what() << endl;
  }

}


StatusSimplex SimplexSolver::algoritmoPrimeiraEtapa(){

  //variavel para armazenar o resultado da verificacao
  //da permissividade (factivel = regiao permissivel) da solucao inicial
  bool baseNegativaEncontrada = false;
  int linhaBaseNegativo = 0;

  bool elemPositivoEncontrado = false;

  //zerar linha e coluna permissiveis
  this->colunaPerm = -1;
  this->linhaPerm = -1;

  string nomeVarBase;
  vector<string>::iterator itVarBase;
  int posVarBase = -1;
  float valorVarBase = 0;
  float valorVarLinhaComTermoNeg = 0.0;

  //verificar se alguns dos valores da coluna dos termos
  //livres possui valor negativo, se possui, temos que pivotar previamente
  //pois nem estamos na regiao permissivel
  for (int linha = 0; linha < this->quadro->totalLinhas - 1; linha++){

    valorVarBase = recuperarTermoLivreLinha(linha);

    if (valorVarBase < 0)
    {
      baseNegativaEncontrada = true;
      linhaBaseNegativo = linha;
      break;
    }
  }

  //se a base negativa nao foi encontrada, significa que estamos na regiao permissivel
  //e podemos avancar para a segunda etapa do algoritmo
  if (!baseNegativaEncontrada)
    return SegundaEtapa;

  //se ela foi encontrada, entao devemos encontrar a linha e a coluna
  //permissivel para efetuar o pivotamento
  //procurar, na linha contendo a base negativa encontrada
  //um elemento negativo
  for (int i = 0; i < this->quadro->totalColunas - 1; i++){
    valorVarLinhaComTermoNeg = this->quadro->matriz[linhaBaseNegativo * this->quadro->totalColunas + i];
    if (valorVarLinhaComTermoNeg < 0){
      this->colunaPerm = i;
      elemPositivoEncontrado = true;
      break;
    }
  }

  //se nao encontrou o elemento positivo,
  //a solucao eh impossivel
  if (!elemPositivoEncontrado)
    return SolucaoImpossivel;

  this->calcularLinhaPermissivel();

  return AlgoritmoTroca;
}

bool SimplexSolver::colunaPertenceABase(int coluna){
  ///verificar se o header apontado na coluna,
  //ja se encontra na base (rowHeader)
  string nomeVarColuna = this->quadro->colHeader[coluna];
  vector<string>::iterator itVarBase = std::find(this->quadro->rowHeader.begin(), this->quadro->rowHeader.end(), nomeVarColuna);
  bool pertenceABase = itVarBase != this->quadro->rowHeader.end();

  return pertenceABase;
}

void SimplexSolver::calcularLinhaPermissivel(){


  float razaoAux = 0.0;
  float menorQuociente = FLT_MAX;
  float eleColunaPerm = 0.0;

  //de posse da coluna permissivel
  //calcular a menor razao entre os elementos da coluna permissivel
  //e designa-la como a linha permissivel
  // o -1 significa para nao computar a ultima linha do quadro, pois essa eh a linha da FO
  for (int linha = 0; linha < this->quadro->totalLinhas - 1; linha++){
    eleColunaPerm = this->quadro->matriz[linha * this->quadro->totalColunas + this->colunaPerm];
    if (eleColunaPerm == 0)
      continue;
    razaoAux = recuperarTermoLivreLinha(linha) / eleColunaPerm;
    if (razaoAux < menorQuociente && razaoAux > 0){
      menorQuociente = razaoAux;
      this->linhaPerm = linha;
    }
  }

  if (this->linhaPerm == -1){
    cout << endl << "Ocorreu um erro para determinar a linha permissivel" << endl;
    throw - 10;
  }

}

float SimplexSolver::recuperarTermoLivreLinha(int linha){
  return this->quadro->matriz[linha * this->quadro->totalColunas + this->quadro->totalColunas - 1];
}

StatusSimplex SimplexSolver::algoritmoSegundaEtapa(){

  bool valorNegativoEncontrado = false;
  float auxValorUltimaLinha = 0.0;
  float maisNegativo = 0;

  //procurar pelo valor mais negativo
  //na linha da FO, ou seja, na ultima linha
  for (int coluna = 0; coluna < this->quadro->totalColunas - 2; coluna++){
    auxValorUltimaLinha = this->quadro->matriz[(this->quadro->totalLinhas - 1) * this->quadro->totalColunas + coluna];
    if (auxValorUltimaLinha < maisNegativo)
    {
      maisNegativo = auxValorUltimaLinha;
      this->colunaPerm = coluna;
      valorNegativoEncontrado = true;
    }
  }

  //se nao foi encontrado valor negativo, significa
  //que a solucao atual eh otima
  if (!valorNegativoEncontrado)
    return SolucaoOtima;

  this->calcularLinhaPermissivel();

  return AlgoritmoTroca;
}

StatusSimplex SimplexSolver::algoritmoTroca(){

  //de posse da linha e coluna permissiveis
  //multiplicar toda a linha do elemento permissivel pelo seu inverso
  float epInverso = 1 / this->quadro->matriz[this->linhaPerm * this->quadro->totalColunas + this->colunaPerm];
  float fatorAnulador = 0.0;

  for (int coluna = 0; coluna < this->quadro->totalColunas; coluna++){
    this->quadro->matriz[this->linhaPerm * this->quadro->totalColunas + coluna] =
      this->quadro->matriz[this->linhaPerm * this->quadro->totalColunas + coluna] * epInverso;
  }

  //usando a linha permissivel calculada,
  //satisfazer a equacao, para deixar todas os elementos da coluna permissivel
  //iguais a 0, ou seja, percorrer toda a matriz para isso
  for (int linha = 0; linha < this->quadro->totalLinhas; linha++)
    //processar somente as demais linhas, sem ser a permissivel
    if (linha != this->linhaPerm){
      //calcular coeficiente que anula o elemento da coluna permissivel da linha atual
      fatorAnulador = this->quadro->matriz[linha * this->quadro->totalColunas + this->colunaPerm] * (-1);

      for (int coluna = 0; coluna < this->quadro->totalColunas; coluna++){
        //o valor da coluna permissivel sera 0
        if (coluna == this->colunaPerm)
          this->quadro->matriz[linha * this->quadro->totalColunas + coluna] = 0;
        else
          //os demais valores devem respeitar a equacao
          //Valor = FatorAnulador * ValorLinhaPerm + LinhaAtual;
          this->quadro->matriz[linha * this->quadro->totalColunas + coluna] = fatorAnulador *
          this->quadro->matriz[this->linhaPerm * this->quadro->totalColunas + coluna] +
          this->quadro->matriz[linha * this->quadro->totalColunas + coluna];
      }
    }


  //trocar headers
  this->quadro->rowHeader[this->linhaPerm] = this->quadro->colHeader[this->colunaPerm];

  //retornar a primeira etapa para sabermos se estamos na regiao permissivel
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
  return this->quadro->matriz[this->quadro->totalColunas * this->quadro->totalLinhas - 1];
}