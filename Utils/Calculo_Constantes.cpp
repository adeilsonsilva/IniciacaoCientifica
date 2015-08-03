#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atof */
#include <math.h> 

#include <signal.h>
#include <opencv2/opencv.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>


using namespace std;
using namespace cv;

const string PATH_CSV = "/home/matheusm/libfreenect2/examples/protonect/csvConstantes.txt";

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, char *argv[])
{
	vector<Mat> images;
	vector<int> labels;
	vector<double> verdadeiros, falsos;
	double somaVerdadeiros=0, somaFalsos=0;
	try {
		read_csv(PATH_CSV, images, labels);
	} catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    for(int i=0; i<images.size(); i += 5) {
    	vector<Mat> training;
        vector<int> labelsTreino;

    	training.push_back(images[i]);
        training.push_back(images[i+1]);
        training.push_back(images[i+2]);
        training.push_back(images[i+3]);
        training.push_back(images[i+4]);
        labelsTreino.push_back(labels[i]);
        labelsTreino.push_back(labels[i]);
        labelsTreino.push_back(labels[i]);
        labelsTreino.push_back(labels[i]);
        labelsTreino.push_back(labels[i]);

    	model->train(training, labelsTreino);
    	for(int j=0; j<images.size(); j++) {
            if(j < i || j > i + 4) {
        		double confidence;
                int prediction;
                Mat teste = images[j];
                model->predict(teste, prediction, confidence);
                if(labels[j]==labels[i]) {
                    verdadeiros.push_back(confidence);
                    somaVerdadeiros += confidence;
                }
                else {
                    falsos.push_back(confidence);
                    somaFalsos += confidence;
                }
            }
    	}
    }
    float variancia = 0;
    float media = somaVerdadeiros/verdadeiros.size();
    for(int i = 0; i < verdadeiros.size(); i++) {
        variancia = variancia + (pow(verdadeiros[i] - media,2));
    }
    variancia /= verdadeiros.size();
    cout << "Media dos Verdadeiros: " << media << endl;
    cout << "Desvio Padrao dos Verdadeiros: " << sqrt(variancia) << endl;
    cout << "------------------------------------------------" << endl;
    media = somaFalsos/falsos.size();
    variancia = 0;
    for(int i = 0; i < falsos.size(); i++)
        variancia = variancia + (pow(falsos[i] - media,2));
    variancia /= falsos.size();
    cout << "Media dos Falsos: " << media << endl;
    cout << "Desvio Padrao dos Falsos: " << sqrt(variancia) << endl;

	return 0;
}
