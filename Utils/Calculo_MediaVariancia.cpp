#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atof */
#include <math.h> 


using namespace std;

int main(int argc, char *argv[])
{
	string filename = argv[1];
	std::ifstream file(filename.c_str(), ifstream::in);
	string line;
	vector<float> valores;
	float soma = 0, variancia = 0;
    while (getline(file, line)) {
    	float f = atof(line.c_str());
    	valores.push_back(f);
    	soma += f;
    }
    float media = soma/valores.size();
    for(int i = 0; i < valores.size(); i++)
        variancia = (variancia + (pow(valores[i] - media,2))/valores.size());
    cout << "Media: " << media << endl;
    cout << "Desvio Padrao: " << sqrt(variancia) << endl;

	return 0;
}