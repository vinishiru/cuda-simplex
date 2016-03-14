#ifndef __STOPWATCH_H_INCLUDED__
#define __STOPWATCH_H_INCLUDED__

#include <chrono>

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
  std::chrono::system_clock::time_point begin;
  std::chrono::system_clock::time_point end;

};

#endif