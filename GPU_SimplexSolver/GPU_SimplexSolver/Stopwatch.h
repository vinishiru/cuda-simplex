#ifndef __STOPWATCH_H_INCLUDED__
#define __STOPWATCH_H_INCLUDED__

#include <ctime>

/*
Classe auxiliar para medir os tempos
dos algoritmos.
**/
class Stopwatch {

public:
  void Start();
  void Stop();

  double Elapsed();

private:
  clock_t begin;
  clock_t end;

};

#endif