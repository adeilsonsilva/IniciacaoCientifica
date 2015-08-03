#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>

const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.45;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;     //0.80
const double DESIRED_LEFT_EYE_X = 0.26;     // Controls how much of the face is visible after preprocessing. 0.16 and 0.14
const double DESIRED_LEFT_EYE_Y = 0.24;
const int MAX_FRAME_LOST = 4;
const int BORDER = 8;  // Border between GUI elements to the edge of the image.
const int FACE_SIZE = 200;
//parametros reconhecimento continuo
const float Usafe = 30.0509; //42.1643;
const float UnotSafe = 95.6884; //109.791;
const float Rsafe = 24.8666; //35.482; 
const float RnotSafe = 30.7036; //41.2861; 

const float E = 2.71828182845904523536;
const float LN2 = 0.693147180559945309417;
const float K_DROP = 15.0;
const int FRAMES_LOGIN = 5;

using namespace cv;
using namespace cv::face;
using namespace std;

const string PATH_CASCADE_FACE = "/home/matheusm/Cascades/IR_Cascade.xml";
const string PATH_CASCADE_RIGHTEYE = "/home/matheusm/Cascades/haarcascade_mcs_righteye_alt.xml";
const string PATH_CASCADE_LEFTEYE = "/home/matheusm/Cascades/haarcascade_mcs_lefteye_alt.xml";
const string PATH_CASCADE_BOTHEYES = "/home/matheusm/Cascades/haarcascade_eye.xml";
//const string PATH_CSV_FACES = "/home/matheusm/framesVideoSujeito4.txt";

int framesLost = 0;
KalmanFilter KFrightEye, KFleftEye;
Mat_<float> state; /* (x, y, Vx, Vy) */
Mat processNoise;
Mat_<float> measurement;
bool initiKalmanFilter = true;
Point leftEyePredict = Point(-1,-1);
Point rightEyePredict = Point(-1,-1);

extern "C" void __cxa_pure_virtual(void)
{
   std::cout << "__cxa_pure_virtual: pure virtual method called" << std::endl << std::flush;
}

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
  int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  

  Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
  Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

  CascadeClassifier haar_cascade;
  haar_cascade.load(PATH_CASCADE_LEFTEYE);

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
    haar_cascade.load(PATH_CASCADE_BOTHEYES);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedLeftEye);
    for(int i = 0; i < detectedLeftEye.size(); i++) {
      Rect eye_i = detectedLeftEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      leftEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    }
  }
  //Find the right eye
  haar_cascade.load(PATH_CASCADE_RIGHTEYE);
  haar_cascade.detectMultiScale(topRightOfFace, detectedRightEye);
  for(int i = 0; i < detectedRightEye.size(); i++) {
    Rect eye_i = detectedRightEye[i];
    eye_i.x += rightX;;
    eye_i.y += topY;
    rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
  }
  // If cascade fails, try another
  if(detectedRightEye.empty()) {
    haar_cascade.load(PATH_CASCADE_BOTHEYES);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedRightEye);
    for(int i = 0; i < detectedRightEye.size(); i++) {
      Rect eye_i = detectedRightEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    }
  }
  //	Inicializacao dos kalman filters
  if(initiKalmanFilter && leftEye.x >= 0 && rightEye.x >= 0) {
    KFrightEye.statePre.at<float>(0) = rightEye.x;
    KFrightEye.statePre.at<float>(1) = rightEye.y;
    KFrightEye.statePre.at<float>(2) = 0;
    KFrightEye.statePre.at<float>(3) = 0;
    KFrightEye.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
    setIdentity(KFrightEye.measurementMatrix);
    setIdentity(KFrightEye.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KFrightEye.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KFrightEye.errorCovPost, Scalar::all(.1));

    KFleftEye.statePre.at<float>(0) = leftEye.x;
    KFleftEye.statePre.at<float>(1) = leftEye.y;
    KFleftEye.statePre.at<float>(2) = 0;
    KFleftEye.statePre.at<float>(3) = 0;
    KFleftEye.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
    setIdentity(KFleftEye.measurementMatrix);
    setIdentity(KFleftEye.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KFleftEye.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KFleftEye.errorCovPost, Scalar::all(.1));
    initiKalmanFilter = false;
  }
  //	Predicao e correcao dos kalman filter
  if(!initiKalmanFilter && leftEye.x >= 0) {
    Mat prediction = KFleftEye.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
    measurement(0) = leftEye.x;
    measurement(1) = leftEye.y;
      
    Point measPt(measurement(0),measurement(1));
    Mat estimated = KFleftEye.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));
    leftEyePredict.x = statePt.x;
    leftEyePredict.y = statePt.y;
  }
  if(!initiKalmanFilter && rightEye.x >= 0) {
    Mat prediction = KFrightEye.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
    measurement(0) = rightEye.x;
    measurement(1) = rightEye.y;
      
    Point measPt(measurement(0),measurement(1));
    Mat estimated = KFrightEye.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));
    rightEyePredict.x = statePt.x;
    rightEyePredict.y = statePt.y;
  }
  //if both eyes were detected
  Mat warped;
  if (leftEye.x >= 0 && rightEye.x >= 0 && ((rightEye.x - leftEye.x) > (face.cols/4))) {
    sucess = true;
    framesLost = 0;
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
  	framesLost++;
  	if(framesLost < MAX_FRAME_LOST && (leftEyePredict.x >= 0 || rightEyePredict.x >= 0)) {
  		if(leftEye.x < 0) {
  			leftEye.x = leftEyePredict.x;
  			leftEye.y = leftEyePredict.y;
  		}
  		if(rightEye.x < 0) {
  			rightEye.x = rightEyePredict.x;
  			rightEye.y = rightEyePredict.y;
  		}
  		if((rightEye.x - leftEye.x) > (face.cols/4)) {
	  		sucess = true;
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
	    	sucess = false;
			return face;
	    }
  	}
  	else {
		sucess = false;
		return face;
  	}
  }
  bilateralFilter(warped, faceProcessed, 0, 20.0, 2.0);
  im_height = faceProcessed.rows;
  Mat mask = Mat(faceProcessed.size(), CV_8U, Scalar(0)); // Start with an empty mask.
  Point faceRect = Point( im_height/2, cvRound(im_height * FACE_ELLIPSE_CY) );
  Size size = Size( cvRound(im_height * FACE_ELLIPSE_W), cvRound(im_height * FACE_ELLIPSE_H) );
  ellipse(mask, faceRect, size, 0, 0, 360, Scalar(255), CV_FILLED);
  // Use the mask, to remove outside pixels.
  Mat dstImg = Mat(faceProcessed.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
  faceProcessed.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
  return dstImg;
}

 void read_csv(const string& filename, vector<Mat>& images, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        if(!path.empty()) {
            images.push_back(imread(path, 0));
        }
    }
}

int main(int argc, char *argv[])
{
	//KALMAN
	measurement.setTo(Scalar(0));
	measurement = Mat_<float>(2,1); 
	KFrightEye = KalmanFilter(4, 2, 0);
	KFleftEye = KalmanFilter(4, 2, 0);
	state = Mat_<float>(4, 1); 
	processNoise = Mat(4, 1, CV_32F);
	// Reconhecimento continuo
	float P_Safe_Ultimo = 1.0, P_notSafe_Ultimo = 0.0;
	float P_Atual_Safe, P_Atual_notSafe;
	float P_Safe_Atual, P_notSafe_Atual;
	float P_Safe;
	float timeLastObs = 0, timeAtualObs = 0, tempo = 0;


    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<Mat> video;
    vector<int> labels;
    /*try {
        read_csv(PATH_CSV_FACES, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << PATH_CSV_FACES << "\". Reason: " << e.msg << endl;
        exit(1);
    }*/
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    //int im_width = images[0].cols;
    //int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    //model->train(images, labels);
    string path = argv[1];
    try {
        read_csv(path, video);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << path << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE);
    bool sucess = false;
    int login = 0;
    Point ultimaFace = Point(-1, -1);
    for(int i = 0; i < video.size(); i++)
    {
    	   Mat frame = video[i];
        //frame.convertTo(frame, CV_8UC3, 255, 0);
        vector< Rect_<int> > faces;
        // Find the faces in the frame:
        haar_cascade.detectMultiScale(frame, faces);
        tempo = getTickCount();
        for(int j = 0; j < faces.size(); j++) {
         	    Mat face = frame(faces[j]);
		    Rect faceRect = faces[j];
		    Point center = Point(faceRect.x + faceRect.width/2, faceRect.y + faceRect.width/2);
		    if(ultimaFace.x < 0) {
		    		ultimaFace.x = center.x;
		    		ultimaFace.y = center.y;
		    }
		    else {
		    		double res = cv::norm(ultimaFace-center);
		    		if(res < 50.0) {
		    			ultimaFace.x = center.x;
			    		ultimaFace.y = center.y;
			    		Mat faceNormalized = faceNormalize(face, FACE_SIZE, sucess);
			          if(sucess) {
			                if(login < FRAMES_LOGIN) {
			                  images.push_back(faceNormalized);
			                  labels.push_back(0);
			                  login++;
			                  if(login == FRAMES_LOGIN) 
			                    model->train(images, labels);
			                }
			                else {
			                    timeLastObs= timeAtualObs;
			                    timeAtualObs = tempo;
			                    double confidence;
			                    int prediction;
			                    model->predict(faceNormalized, prediction, confidence);
			                    //reconhecimento continuo
			                    if(timeLastObs == 0) {
			                    	P_Atual_Safe = 1 - ((1 + erf((confidence-Usafe) / (Rsafe*sqrt(2))))/2);
			                    	P_Atual_notSafe = ((1 + erf((confidence-UnotSafe) / (RnotSafe*sqrt(2))))/2);
			                        float deltaT = (timeLastObs - timeAtualObs)/getTickFrequency();
			                        float elevado = (deltaT * LN2)/K_DROP;
			                    	P_Safe_Atual = P_Atual_Safe + pow(E,elevado) * P_Safe_Ultimo;
			              		    P_notSafe_Atual = P_Atual_notSafe + pow(E,elevado) * P_notSafe_Ultimo;
			                    }
			                    else {
			                    	P_Atual_Safe = 1 - ((1 + erf((confidence-Usafe) / (Rsafe*sqrt(2))))/2);
			                    	P_Atual_notSafe = ((1 + erf((confidence-UnotSafe) / (RnotSafe*sqrt(2))))/2);
			                    	P_Safe_Ultimo = P_Safe_Atual;
			                    	P_notSafe_Ultimo = P_notSafe_Atual;
			                    	float deltaT = (timeLastObs - timeAtualObs)/getTickFrequency();
			                      float elevado = (deltaT * LN2)/K_DROP;
			                      P_Safe_Atual = P_Atual_Safe + pow(E,elevado) * P_Safe_Ultimo;
			                      P_notSafe_Atual = P_Atual_notSafe + pow(E,elevado) * P_notSafe_Ultimo;
			                    }
			                }
			            }
				 }
		    }

              
        }
        if(P_Safe_Atual != 0) {
          float deltaT = -(tempo - timeAtualObs)/getTickFrequency();
          float elevado = (deltaT * LN2)/K_DROP;
          P_Safe = (pow(E,elevado) * P_Safe_Atual)/(P_Safe_Atual+P_notSafe_Atual);
          if(P_Safe > 0)
            cout << P_Safe << endl;
        }

    }
    return 0;
}
