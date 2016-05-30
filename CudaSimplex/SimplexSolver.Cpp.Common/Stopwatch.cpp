#include "stdafx.h"

void Stopwatch::Start(){
  this->begin = std::chrono::high_resolution_clock::now();
}

void Stopwatch::Stop(){
  this->end = std::chrono::high_resolution_clock::now();
}

double Stopwatch::Parcial(){
  std::chrono::system_clock::time_point parcial = std::chrono::high_resolution_clock::now();
  return (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(parcial - begin).count()) / 1000000000;
}

double Stopwatch::Elapsed(){
  return (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000000000;
}