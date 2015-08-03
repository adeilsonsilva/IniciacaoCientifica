#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.45;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;     //0.80
const double DESIRED_LEFT_EYE_X = 0.26;     // Controls how much of the face is visible after preprocessing. 0.16 and 0.14
const double DESIRED_LEFT_EYE_Y = 0.24;

using namespace cv;
using namespace std;

Mat faceNormalize(Mat face, int desiredFaceWidth, bool &sucess) {
  int im_height = face.rows;
  Mat faceProcessed;

  cv::resize(face, face, Size(400, 400), 1.0, 1.0, INTER_CUBIC);

  const float EYE_SX = 0.16f;
  const float EYE_SY = 0.26f;
  const float EYE_SW = 0.30f;
  const float EYE_SH = 0.28f;

  int leftX = cvRound(face.cols * EYE_SX);
  int topY = cvRound(face.rows * EYE_SY);
  int widthX = cvRound(face.cols * EYE_SW);
  int heightY = cvRound(face.rows * EYE_SH);
  int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner

  Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
  Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

  string lsLeftEye_haar = "/home/stremens/Cascades/haarcascade_mcs_lefteye_alt.xml";
  string lsRightEye_haar = "/home/stremens/Cascades/haarcascade_mcs_righteye_alt.xml";
  string lsBothEyes_haar = "/home/stremens/Cascades/haarcascade_eye.xml";

  CascadeClassifier haar_cascade;
  haar_cascade.load(lsLeftEye_haar);

  vector< Rect_<int> > detectedRightEye;
  vector< Rect_<int> > detectedLeftEye;
  Point leftEye = Point(-1, -1), rightEye = Point(-1, -1);
  // Find the left eye:
  haar_cascade.detectMultiScale(topLeftOfFace, detectedLeftEye);
  for(int i = 0; i < detectedLeftEye.size(); i++) {
    Rect eye_i = detectedLeftEye[i];
    eye_i.x += leftX;
    eye_i.y += topY;
    leftEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
  }
  // If cascade fails, try another
  if(detectedLeftEye.empty()) {
    haar_cascade.load(lsBothEyes_haar);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedLeftEye);
    for(int i = 0; i < detectedLeftEye.size(); i++) {
      Rect eye_i = detectedLeftEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      leftEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    }
  }
  haar_cascade.load(lsRightEye_haar);
  haar_cascade.detectMultiScale(topRightOfFace, detectedRightEye);
  for(int i = 0; i < detectedRightEye.size(); i++) {
    Rect eye_i = detectedRightEye[i];
    eye_i.x += rightX;;
    eye_i.y += topY;
    rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
  }
  if(detectedRightEye.empty()) {
    haar_cascade.load(lsBothEyes_haar);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedRightEye);
    for(int i = 0; i < detectedRightEye.size(); i++) {
      Rect eye_i = detectedRightEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    }
  }
  //if both eyes were detected
  Mat warped;
  if (leftEye.x >= 0 && rightEye.x >= 0 && ((rightEye.x - leftEye.x) > (face.cols/4))) {
    sucess = true;
    cout << "Olhos detectados" << endl;
    int desiredFaceHeight = desiredFaceWidth;
    // Get the center between the 2 eyes.
    Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
    // Get the angle between the 2 eyes.
    double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy, dx) * 180.0/CV_PI; // Convert from radians to degrees.

    // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
    const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
    // Get the amount we need to scale the image to be the desired fixed size we want.
    double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
    double scale = desiredLen / len;
    // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
    // Shift the center of the eyes to be the desired center between the eyes.
    rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
    rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

    // Rotate and scale and translate the image to the desired angle & size & position!
    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
    warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
    warpAffine(face, warped, rot_mat, warped.size());
    equalizeHist(warped, warped);
  }
  else {
  	cout << "Erro: Olhos não detectados" << endl;
    sucess = false;
    return face;
  }
  bilateralFilter(warped, faceProcessed, 0, 20.0, 2.0);
  im_height = faceProcessed.rows;
  Mat mask = Mat(faceProcessed.size(), CV_8U, Scalar(0)); // Start with an empty mask.
  Point faceCenter = Point( im_height/2, cvRound(im_height * FACE_ELLIPSE_CY) );
  Size size = Size( cvRound(im_height * FACE_ELLIPSE_W), cvRound(im_height * FACE_ELLIPSE_H) );
  ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
  // Use the mask, to remove outside pixels.
  Mat dstImg = Mat(faceProcessed.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
  faceProcessed.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
  return dstImg;
}

// Funcao que carrega um vetor de imagens a partir de um csv
static void read_csv(const string& filename, vector<Mat>& images, vector<string>& labels, char separator = ';') {
	cout << "Reading csv..." << endl;
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int coutImage = 0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
        	cout << "Loading image " << coutImage << endl;
        	coutImage++;
            images.push_back(imread(path, 0));
            labels.push_back(classlabel);
        }
    }
}

int main(int argc, const char *argv[]) {
	// Verifica se tem a quantidade necessária de argumentos
    if (argc < 1) {
        cout << "usage: " << argv[0] << " <csvFacesDesnormalizadas.ext>" << endl;
        exit(1);
    }
    string lsFacesCSV = string(argv[1]);

    vector<Mat> imagesFaces;
    vector<string> labelsFaces;

    try {
        read_csv(lsFacesCSV, imagesFaces, labelsFaces);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    if(imagesFaces.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work in each set. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
        exit(1);
    }

    for(int i=0; i < imagesFaces.size(); i++) {
    	Mat faceDesnormalizada = imagesFaces[i];
    	bool sucess = false;
    	cout << "Normalizando face numero " << i << endl;
    	Mat faceNormalizada = faceNormalize(faceDesnormalizada, 200, sucess);
    	if(sucess) {
    		cout << "Sucess" << endl;
    		imwrite("/home/stremens/FacesNormalizadas/"+labelsFaces[i]+".pgm", faceNormalizada);
    	}
    }

	return 0;
}