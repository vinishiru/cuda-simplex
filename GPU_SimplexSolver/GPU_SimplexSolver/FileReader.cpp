#include "FileReader.h"


FileReader::FileReader(string filePath){

	mpsFile.open(filePath);
}

FileReader::~FileReader(){
	mpsFile.close();
}

void FileReader::closeFile(){
	mpsFile.close();
}

string FileReader::readLine(){

	string line = "";

	if ( !mpsFile.eof() ){
		getline(mpsFile,line);
	}

	return line;
}