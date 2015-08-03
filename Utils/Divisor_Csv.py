import sys, math, Image, os
import numpy as np

if __name__ == "__main__":
	try:
        	inFile = open('treinados.txt')
    	except:
		raise IOError('There is no file named path_to_created_csv_file.csv in current directory.')

	treinados = []
	teste = []
	train = 1
	contador = 0
	for line in inFile.readlines():
		if contador == 10:
			contador = 0
			train *= -1
		if train == 1:
			treinados.append(line)
		else:
			teste.append(line)
		contador+=1
	
	inFile.close()	
	arquivo = open('treino.txt', 'w')
	arquivo.writelines(treinados)
	arquivo.close()

	arquivo = open('testePositivo.txt', 'w')
	arquivo.writelines(teste)
	arquivo.close()
