#ifndef __STRINGTOKENIZER_H_INCLUDED__
#define __STRINGTOKENIZER_H_INCLUDED__

#include <string>

using namespace std;

class StringTokenizer {

private:
	string line;
	string token;
	int lineIndex;
	int lastLineIndex;


public:
	StringTokenizer(string line);
	~StringTokenizer();

	string nextToken();
	bool hasToken();
	void setLine(string line);
	void returnToLastIndex();
	
};

#endif