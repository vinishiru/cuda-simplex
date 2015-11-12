#ifndef __ILPREADER_H_INCLUDED
#define __ILPREADER_H_INCLUDED

#include "FObjetivo.h"

class ILPReader{

public:
  virtual FObjetivo* LerFuncaoObjetivo() = 0;
};

#endif