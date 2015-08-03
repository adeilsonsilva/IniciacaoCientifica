!/bin/bash
echo 'Criando positives samples'
perl bin/createsamples.pl positives.txt negatives.txt samples 263660\
  "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 0.5\
  -maxyangle 0.5 maxzangle 0.2 -maxidev 40 -w 20 -h 20"

find ./samples -name '*.vec' > samples.txt
echo 'Merging samples'
./mergevec samples.txt samples.vec

echo 'Treining classifier'
opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
  -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 42185\
  -numNeg 53225 -w 20 -h 20 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024 
