#include "stdafx.h"


void SimplexSolver::otimizar(FObjetivo *func){

  this->quadro = new Quadro(func);


  this->quadro->buildQuadro();

  //exibir quadro montado;
  this->quadro->toString();

}
