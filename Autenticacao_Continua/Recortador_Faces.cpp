#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h> 
#include <string>  
#include <opencv2/opencv.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>

#include <vector>

const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.45;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;     //0.80
const double DESIRED_LEFT_EYE_X = 0.26;     // Controls how much of the face is visible after preprocessing. 0.16 and 0.14
const double DESIRED_LEFT_EYE_Y = 0.24;

using namespace cv;
using namespace std;

const string PATH_CASCADE_FACE = "/home/matheusm/libfreenect2/examples/protonect/Cascades/cascade.xml";
int pasta;
// Funcao que carrega um vetor de imagens a partir de um csv
static void read_csv(const string& filename, vector<Mat>& images, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int contador = 0;
    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE);
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        if(!path.empty()) {
            //images.push_back(imread(path, 0));
            Mat image = imread(path, 0);
            vector< Rect_<int> > faces;
            haar_cascade.detectMultiScale(image, faces);
            for(int i = 0; i < faces.size(); i++) {
                Mat face = image(faces[i]);
		if(face.cols > 120 && face.rows > 120) {
                	imwrite(format("/home/matheusm/Positivas/%d/%d.pgm", pasta, contador), face);
                	Rect face_i = faces[i];
                	rectangle(image, face_i, CV_RGB(0, 0, 0), CV_FILLED);
			imwrite(format("/home/matheusm/Negativas/%d/%d.pgm", pasta, contador), image);
			cout << "Recortando imagen numero " << contador << endl;
			contador++;
		}
            }
        }
    }
}

int main(int argc, const char *argv[]) {
	// Verifica se tem a quantidade necessária de argumentos
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csvFacesDesnormalizadas.ext>" << endl;
        exit(1);
    }
    string lsFacesCSV = string(argv[1]);
    string lsPasta = string(argv[2]);
    pasta = atoi(lsPasta.c_str());

    vector<Mat> imagesFaces;

    try {
        read_csv(lsFacesCSV, imagesFaces);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    /*int contador = 0;
    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE);
    for(int i=0; i < imagesFaces.size(); i++) {
        Mat image = imagesFaces[i];
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(image, faces);
        for(int j = 0; i < faces.size(); i++) {
            Mat face = image(faces[j]);
        	imwrite(format("/home/matheusm/FacesRecordatadas/Positivas/%d.pgm", contador), face);
            Rect face_i = faces[j];
            rectangle(image, face_i, CV_RGB(0, 0, 0), CV_FILLED);
            imwrite(format("/home/matheusm/FacesRecordatadas/Negativas/%d.pgm", contador), image);
            contador++;
        }
    }*/
	return 0;
}
