#include "StringTokenizer.h"

StringTokenizer::StringTokenizer(string line){
	this->line = line;
	this->lineIndex = 0;
}

void StringTokenizer::setLine(string line){
	this->line = line;
	this->lineIndex = 0;
	token.clear();
}

string StringTokenizer::nextToken(){

	char ch;
	token.clear();
	lastLineIndex = lineIndex;

	if (lineIndex < line.length()){
		//Ler primeiro caracter
		ch = line[lineIndex];

		//Passar pelos espacos ate encontrar primeiro caracter
		while (lineIndex < line.length() && line[lineIndex] == ' '){
			lineIndex++;
		}

		//Ler todos caracteres ate encontrar o proximo espaco em branco ou fim da linha
		while (lineIndex < line.length() && line[lineIndex] != ' '){
			ch = line[lineIndex];
			token += ch;
			lineIndex++;			
		}
	}

	return token;
}

bool StringTokenizer::hasToken(){

	int auxIndex = lineIndex;

	if ( !this->nextToken().empty() ) {
		lineIndex = auxIndex;
		return true;
	}
	lineIndex = auxIndex;
	return false;
}

void StringTokenizer::returnToLastIndex(){
	lineIndex = lastLineIndex;
}