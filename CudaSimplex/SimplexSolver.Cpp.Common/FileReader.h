#ifndef __FILEREADERUTIL_H_INCLUDED
#define __FILEREADERUTIL_H_INCLUDED

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class FileReader{

private:
	string line;
	string filePath;
	ifstream mpsFile;


public:

	string readLine();
	void closeFile();

	FileReader(string filePath);
	~FileReader();

};

#endif