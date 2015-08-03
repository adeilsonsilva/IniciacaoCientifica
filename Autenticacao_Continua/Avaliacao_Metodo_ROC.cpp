#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

#define MIN_THR 68.0  // Limiar (threshold) minimo

// Funcao que carrega um vetor de imagens a partir de um csv
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

int main(int argc, const char *argv[]) {
	// Verifica se tem a quantidade necessária de argumentos
    if (argc < 3) {
        cout << "usage: " << argv[0] << " <csvTreino.ext> <csvTestePositivo.ext> <csvTesteNegativo.ext> " << endl;
        exit(1);
    }
    // Pega os caminhos para cada CSV.
    string lsTreinoCSV = string(argv[1]);
    string lsTestePositivoCSV = string(argv[2]);
    string lsTesteNegativoCSV = string(argv[3]);
    // Vetores de imagens para treino e testes.
    vector<Mat> imagensTreino;
    vector<int> labelsTreino;

    vector<Mat> imagensTestePositivo;
    vector<int> labelsTestePositivo;

    vector<Mat> imagensTesteNegativo;
    vector<int> labelsTesteNegativo;

    // Carrega os vetores com as imagens de treino e de testes    
    try {
        read_csv(lsTreinoCSV, imagensTreino, labelsTreino);
        read_csv(lsTestePositivoCSV, imagensTestePositivo, labelsTestePositivo);
        read_csv(lsTesteNegativoCSV, imagensTesteNegativo, labelsTesteNegativo);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(imagensTreino.size() <= 1 || imagensTestePositivo.size() <= 1 || imagensTesteNegativo.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work in each set. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
        exit(1);
    }

    cout << "Eigenfaces" << endl;
    //definicao do limiar e inicializacao dos rates
    double threshold = MIN_THR;
    int falseAccept = 0;
    int trueAccept = 0;

    int testLabel;
    int predictedLabel;
    double confidence;

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(imagensTreino, labelsTreino);

    for(int i = 0; i < imagensTestePositivo.size(); i++) {
        Mat testSample = imagensTestePositivo[i];
        model->predict(testSample, predictedLabel, confidence);
        cout << "1 " << confidence << endl;
    }
    vector<double> resultadoImagensNegativasEigenfaces;
    for(int i = 0; i < imagensTesteNegativo.size(); i++) {
        Mat testSample = imagensTesteNegativo[i];
        model->predict(testSample, predictedLabel, confidence);
        cout << "-1 " << confidence << endl;
    }
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------
    cout << "Fisherfaces" << endl;
    model = createFisherFaceRecognizer();
    model->train(imagensTreino, labelsTreino);
    
    for(int i = 0; i < imagensTestePositivo.size(); i++) {
        Mat testSample = imagensTestePositivo[i];
        model->predict(testSample, predictedLabel, confidence);
        cout << "1 " << confidence << endl;
    }
    vector<double> resultadoImagensNegativasFisherfaces;
    for(int i = 0; i < imagensTesteNegativo.size(); i++) {
        Mat testSample = imagensTesteNegativo[i];
        model->predict(testSample, predictedLabel, confidence);
        cout << "-1 " << confidence << endl;
    }
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------
    cout << "LBPH" << endl;
    model = createLBPHFaceRecognizer();
    model->train(imagensTreino, labelsTreino);

    for(int i = 0; i < imagensTestePositivo.size(); i++) {
        Mat testSample = imagensTestePositivo[i];
        model->predict(testSample, predictedLabel, confidence);
        cout << "1 " << confidence << endl;
    }
    vector<double> resultadoImagensNegativasLBPH;
    for(int i = 0; i < imagensTesteNegativo.size(); i++) {
        Mat testSample = imagensTesteNegativo[i];
        model->predict(testSample, predictedLabel, confidence);
        cout << "-1 " << confidence << endl;
    }

	return 0;
}