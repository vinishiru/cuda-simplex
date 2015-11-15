#include "MPSReader.h"
#include "StringTokenizer.h"

MPSReader::MPSReader(std::string mpsPath){
  funcao = new FObjetivo();
  fileReader = new FileReader(mpsPath);
  VetorRHSPossuiNome = true;
}

FObjetivo* MPSReader::LerFuncaoObjetivo(){

  readNAME();
  readROWS();
  readCOLUMNS();
  readRHS();
  //Temos que verificar se realmente sera necessario utilizar as restricoes
  //finais, chamadas BOUNDS do arquivo MPS.
  //readBOUNDS();
  fileReader->closeFile();

  return this->funcao;
}

void MPSReader::readBOUNDS(){

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

  StringTokenizer *lineTokens = new StringTokenizer(line);
  string boundType;
  string boundName;
  string nomeVariavel;
  string boundValue;
  string auxRestName; //sera criada uma nova restricao se a fronteira for valida
  Desigualdade tipoDesigualdade;

  if (lineTokens->nextToken().compare("BOUNDS") == 0){

    line = fileReader->readLine();

    while (line.compare("ENDATA") != 0){

      lineTokens->setLine(line);

      //Ler no minimo 3 tokens
      //Nome variavel / Nome Funcao|Restricao / Valor variavel

      boundType = lineTokens->nextToken();
      boundName = lineTokens->nextToken();
      nomeVariavel = lineTokens->nextToken();
      boundValue = lineTokens->nextToken();

      if (boundType.compare("LO") == 0){
        tipoDesigualdade = MaiorOuIgual;
      }
      else if (boundType.compare("UP") == 0){
        tipoDesigualdade = MenorOuIgual;
      }
      auxRestName = funcao->addRestricao();
      //Configurar nova restricao
      funcao->addVariavelRestricao(auxRestName, nomeVariavel, 1.0);
      funcao->setDesigualdadeRestricao(auxRestName, tipoDesigualdade);
      funcao->setTermoLivreRestricao(auxRestName, stof(boundValue.c_str()));

      line = fileReader->readLine(); //ler nova linha ao final
    }

  }

}

void MPSReader::readRHS(){

  StringTokenizer *lineTokens = new StringTokenizer(line);
  string nomeVetorRHS;
  string nomeFuncao;
  string valorTermoLivre;

  if (lineTokens->nextToken().compare("RHS") == 0){

    line = fileReader->readLine();

    while (line.compare("BOUNDS") != 0 && line.compare("") != 0 && line.compare("ENDATA") != 0){

      lineTokens->setLine(line);

      //Ler no minimo 3 tokens
      //Nome variavel / Nome Funcao|Restricao / Valor variavel

      if (this->VetorRHSPossuiNome){
        nomeVetorRHS = lineTokens->nextToken();
        nomeFuncao = lineTokens->nextToken();
        valorTermoLivre = lineTokens->nextToken();
      }
      else{
        nomeFuncao = lineTokens->nextToken();
        valorTermoLivre = lineTokens->nextToken();
      }

      //Se nomeFuncao for o nome da funcao objetivo, atribuir variaveis nela
      if (nomeFuncao.compare(funcao->Nome) == 0){
        funcao->TermoLivre = stof(valorTermoLivre.c_str());
      }
      //Se nao, atribuir variaveis a restricao
      else {
        funcao->setTermoLivreRestricao(nomeFuncao, stof(valorTermoLivre.c_str()));
      }

      //Se tiver mais tokens, ler mais duas colunas
      //Nome Variavel e Valor Variavel, obedecendo as mesmas regras anteriores
      if (lineTokens->hasToken()){

        nomeFuncao = lineTokens->nextToken();
        valorTermoLivre = lineTokens->nextToken();

        if (nomeFuncao.compare(funcao->Nome) == 0){
          funcao->TermoLivre = stof(valorTermoLivre.c_str());
        }
        //Se nao, atribuir variaveis a restricao
        else {
          funcao->setTermoLivreRestricao(nomeFuncao, stof(valorTermoLivre.c_str()));
        }
      }

      line = fileReader->readLine(); //ler nova linha ao final
    }

  }
}

void MPSReader::readCOLUMNS(){

  StringTokenizer *lineTokens = new StringTokenizer(line);
  string nomeFuncao;
  string nomeVariavel;
  string valorVariavel;

  if (lineTokens->nextToken().compare("COLUMNS") == 0){

    line = fileReader->readLine();

    while (line.compare("RHS") != 0){

      lineTokens->setLine(line);

      //Ler no minimo 3 tokens
      //Nome variavel / Nome Funcao|Restricao / Valor variavel

      nomeVariavel = lineTokens->nextToken();
      nomeFuncao = lineTokens->nextToken();
      valorVariavel = lineTokens->nextToken();

      //Se nomeFuncao for o nome da funcao objetivo, atribuir variaveis nela
      if (nomeFuncao.compare(funcao->Nome) == 0){
        funcao->addVariavel(nomeVariavel, stof(valorVariavel.c_str()));
      }
      //Se nao, atribuir variaveis a restricao
      else {
        funcao->addVariavelRestricao(nomeFuncao, nomeVariavel, stof(valorVariavel.c_str()));
      }

      //Se tiver mais tokens, ler mais duas colunas
      //Nome Variavel e Valor Variavel, obedecendo as mesmas regras anteriores
      if (lineTokens->hasToken()){
        nomeFuncao = lineTokens->nextToken();
        valorVariavel = lineTokens->nextToken();

        //Se nomeFuncao for o nome da funcao objetivo, atribuir variaveis nela
        if (nomeFuncao.compare(funcao->Nome) == 0){
          funcao->addVariavel(nomeVariavel, stof(valorVariavel.c_str()));
        }
        //Se nao, atribuir variaveis a restricao
        else {
          funcao->addVariavelRestricao(nomeFuncao, nomeVariavel, stof(valorVariavel.c_str()));
        }
      }


      line = fileReader->readLine(); //ler nova linha ao final
    }
  }

}

void MPSReader::readROWS(){

  line = fileReader->readLine();
  StringTokenizer *lineTokens = new StringTokenizer(line);
  string desigualdade;
  string nomeFuncao;

  if (lineTokens->nextToken().compare("ROWS") == 0){

    line = fileReader->readLine();

    while (line.compare("COLUMNS") != 0){

      lineTokens->setLine(line);

      desigualdade = lineTokens->nextToken();
      nomeFuncao = lineTokens->nextToken();

      //Objetivo
      if (desigualdade.compare("N") == 0){
        funcao->Nome = nomeFuncao;
      }
      //Igualdade
      else if (desigualdade.compare("E") == 0){
        funcao->addRestricao(nomeFuncao);
        funcao->setDesigualdadeRestricao(nomeFuncao, Igual);
      }
      //Menor ou igual
      else if (desigualdade.compare("L") == 0){
        funcao->addRestricao(nomeFuncao);
        funcao->setDesigualdadeRestricao(nomeFuncao, MenorOuIgual);
      }
      //Maior ou igual
      else if (desigualdade.compare("G") == 0){
        funcao->addRestricao(nomeFuncao);
        funcao->setDesigualdadeRestricao(nomeFuncao, MaiorOuIgual);
      }
      else
        throw 10; //Erro de sintaxe

      line = fileReader->readLine();
    }
  }


}

void MPSReader::readNAME(){

  line = fileReader->readLine();
  StringTokenizer *lineTokens = new StringTokenizer(line);

  if (lineTokens->nextToken().compare("NAME") == 0)
  {
    funcao->NomeProblema = lineTokens->nextToken();
  }
}
