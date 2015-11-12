#include "Stopwatch.h"

void Stopwatch::Start(){
  this->begin = clock();
}

void Stopwatch::Stop(){
  this->end = clock();
}

double Stopwatch::Elapsed(){
  return double(this->end - this->begin) / CLOCKS_PER_SEC;
}