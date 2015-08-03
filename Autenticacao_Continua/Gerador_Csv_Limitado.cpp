#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
const string PATH = "/home/matheusm/Record/s4/";

int main(int argc, char *argv[])
{
	ofstream myfile;
  	myfile.open ("framesVideoSujeito4.txt");
  	for(int i = 1; i <= 1005; i++)
  		myfile << PATH << "frame" << i << ".pgm"<< endl;
  	myfile.close();


	return 0;
}
