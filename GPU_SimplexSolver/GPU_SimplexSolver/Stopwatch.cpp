#include "Stopwatch.h"

void Stopwatch::Start(){
  this->begin = std::chrono::high_resolution_clock::now();
}

void Stopwatch::Stop(){
  this->end = std::chrono::high_resolution_clock::now();
}

double Stopwatch::Elapsed(){
  return (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000000000;
}