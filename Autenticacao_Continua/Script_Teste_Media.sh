#!/bin/sh
g++ Calculo_Variancia_Dinamico.cpp `pkg-config --cflags --libs opencv`
echo '1x2'
./a.out /home/matheusm/Record/1x2.txt > m1x2.txt
echo '1x3'
./a.out /home/matheusm/Record/1x3.txt > m1x3.txt
echo '1x4'
./a.out /home/matheusm/Record/1x4.txt > m1x4.txt
echo '2x1'
./a.out /home/matheusm/Record/2x1.txt > m2x1.txt
echo '2x3'
./a.out /home/matheusm/Record/2x3.txt > m2x3.txt
echo '2x4'
./a.out /home/matheusm/Record/2x4.txt > m2x4.txt
echo '3x1'
./a.out /home/matheusm/Record/3x1.txt > m3x1.txt
echo '3x2'
./a.out /home/matheusm/Record/3x2.txt > m3x2.txt
echo '3x4'
./a.out /home/matheusm/Record/3x4.txt > m3x4.txt
echo '4x1'
./a.out /home/matheusm/Record/4x1.txt > m4x1.txt
echo '4x2'
./a.out /home/matheusm/Record/4x2.txt > m4x2.txt
echo '4x3'
./a.out /home/matheusm/Record/4x3.txt > m4x3.txt
