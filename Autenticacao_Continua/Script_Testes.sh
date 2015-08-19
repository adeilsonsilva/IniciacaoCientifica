!/bin/bash
g++ Teste_Video_PSafe.cpp `pkg-config --cflags --libs opencv`
echo '1'
./a.out /home/matheusm/Record/framesVideoSujeito1.txt > sujeito1.txt
echo '2'
./a.out /home/matheusm/Record/framesVideoSujeito2.txt > sujeito2.txt
echo '3'
./a.out /home/matheusm/Record/framesVideoSujeito3.txt > sujeito3.txt
echo '4'
./a.out /home/matheusm/Record/framesVideoSujeito4.txt > sujeito4.txt
echo '1x2'
./a.out /home/matheusm/Record/1x2.txt > p1x2.txt
echo '1x3'
./a.out /home/matheusm/Record/1x3.txt > p1x3.txt
echo '1x4'
./a.out /home/matheusm/Record/1x4.txt > p1x4.txt
echo '2x1'
./a.out /home/matheusm/Record/2x1.txt > p2x1.txt
echo '2x3'
./a.out /home/matheusm/Record/2x3.txt > p2x3.txt
echo '2x4'
./a.out /home/matheusm/Record/2x4.txt > p2x4.txt
echo '3x1'
./a.out /home/matheusm/Record/3x1.txt > p3x1.txt
echo '3x2'
./a.out /home/matheusm/Record/3x2.txt > p3x2.txt
echo '3x4'
./a.out /home/matheusm/Record/3x4.txt > p3x4.txt
echo '4x1'
./a.out /home/matheusm/Record/4x1.txt > p4x1.txt
echo '4x2'
./a.out /home/matheusm/Record/4x2.txt > p4x2.txt
echo '4x3'
./a.out /home/matheusm/Record/4x3.txt > p4x3.txt
