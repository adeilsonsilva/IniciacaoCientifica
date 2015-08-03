/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
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
using namespace std;

const string PATH_CASCADE_FACE = "/home/matheusm/libfreenect2/examples/protonect/Cascades/cascade.xml";
const string PATH_CASCADE_RIGHTEYE = "/home/matheusm/libfreenect2/examples/protonect/Cascades/haarcascade_mcs_righteye_alt.xml";
const string PATH_CASCADE_LEFTEYE = "/home/matheusm/libfreenect2/examples/protonect/Cascades/haarcascade_mcs_lefteye_alt.xml";
const string PATH_CASCADE_BOTHEYES = "/home/matheusm/libfreenect2/examples/protonect/Cascades/haarcascade_eye.xml";

bool protonect_shutdown = false;

void sigint_handler(int s)
{
	protonect_shutdown = true;
}

int framesLost = 0;
KalmanFilter KFrightEye, KFleftEye;
Mat_<float> state; /* (x, y, Vx, Vy) */
Mat processNoise;
Mat_<float> measurement;
bool initiKalmanFilter = true;
Point leftEyePredict = Point(-1,-1);
Point rightEyePredict = Point(-1,-1);

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
	//	Find the right eye
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
		KFrightEye.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
		setIdentity(KFrightEye.measurementMatrix);
		setIdentity(KFrightEye.processNoiseCov, Scalar::all(1e-4));
		setIdentity(KFrightEye.measurementNoiseCov, Scalar::all(1e-1));
		setIdentity(KFrightEye.errorCovPost, Scalar::all(.1));

		KFleftEye.statePre.at<float>(0) = leftEye.x;
		KFleftEye.statePre.at<float>(1) = leftEye.y;
		KFleftEye.statePre.at<float>(2) = 0;
		KFleftEye.statePre.at<float>(3) = 0;
		KFleftEye.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
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
	Point faceCenter = Point( im_height/2, cvRound(im_height * FACE_ELLIPSE_CY) );
	Size size = Size( cvRound(im_height * FACE_ELLIPSE_W), cvRound(im_height * FACE_ELLIPSE_H) );
	ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
	// Use the mask, to remove outside pixels.
	Mat dstImg = Mat(faceProcessed.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
	faceProcessed.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
	return dstImg;
}

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

	std::string program_path(argv[0]);
	size_t executable_name_idx = program_path.rfind("Protonect");
	std::string binpath = "/";
	if(executable_name_idx != std::string::npos)
	{
		binpath = program_path.substr(0, executable_name_idx);
	}
	libfreenect2::Freenect2 freenect2;
	libfreenect2::Freenect2Device *dev = freenect2.openDefaultDevice();
	if(dev == 0)
	{
		std::cout << "no device connected or failure opening the default one!" << std::endl;
		return -1;
	}
	signal(SIGINT,sigint_handler);
	protonect_shutdown = false;
	libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
	libfreenect2::FrameMap frames;
	dev->setColorFrameListener(&listener);
	dev->setIrAndDepthFrameListener(&listener);
	dev->start();
	std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
	std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

	vector<Mat> images;
	vector<int> labels;
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    CascadeClassifier haar_cascade;
    haar_cascade.load(PATH_CASCADE_FACE);
    bool sucess = false;
    int login = 0;
    while(!protonect_shutdown)
    {
    	listener.waitForNewFrame(frames);
    	libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    	Mat frame = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data)/ 80000;
    	frame.convertTo(frame, CV_8UC3, 255, 0);
    	vector< Rect_<int> > faces;
    	// Find the faces in the frame:
    	haar_cascade.detectMultiScale(frame, faces);

    	tempo = getTickCount();
    	for(int i = 0; i < faces.size(); i++) {
    		Mat face = frame(faces[i]);
    		Rect face_i = faces[i];
    		Mat faceNormalized = faceNormalize(face, FACE_SIZE, sucess);
    		rectangle(frame, face_i, CV_RGB(0, 0, 255), 1);
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
    				Mat srcBGR = faceNormalized;
    				cv::resize(srcBGR, srcBGR, Size(100, 100), 1.0, 1.0, INTER_CUBIC);
    				int cx = (frame.cols - srcBGR.cols) - BORDER;
    				Rect dstRC = Rect(cx, BORDER, srcBGR.cols, srcBGR.cols);
    				Mat dstROI = frame(dstRC);
                	// Copy the pixels from src to dst.
    				srcBGR.copyTo(dstROI);
                	// Create the text we will annotate the box with:
    				string box_text = format("Confidence = %d", (int)confidence);
                	// Calculate the position for annotated text (make sure we don't
                	// put illegal values in there):
    				int pos_x = std::max(face_i.tl().x - 10, 0);
    				int pos_y = std::max(face_i.tl().y - 10, 0);
    				putText(frame, box_text, Point(face_i.x - box_text.size() + 10, face_i.y - 5), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(255,255,255), 1.0);
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
    	if(P_Safe_Atual != 0) {
    		float deltaT = -(tempo - timeAtualObs)/getTickFrequency();
    		float elevado = (deltaT * LN2)/K_DROP;
    		P_Safe = (pow(E,elevado) * P_Safe_Atual)/(P_Safe_Atual+P_notSafe_Atual);
    	}

    	if(login == FRAMES_LOGIN) {
    		string pSafe = format("Probability System Safe = %f", P_Safe);
    		putText(frame, pSafe, Point(BORDER, frame.rows - BORDER), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(255,255,255), 1.0);
    	}
    	else {
    		putText(frame, "Login in process...", Point(BORDER, frame.rows - BORDER), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(255,255,255), 1.0);
    	}
    	cv::resize(frame, frame, Size(frame.cols*1.5, frame.rows*1.5), 1.0, 1.0, INTER_CUBIC);
    	cv::imshow("Infrared Face Recognition", frame);
    	int key = cv::waitKey(1);
	    protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape
	    listener.release(frames);
    }

    dev->stop();
    dev->close();

    return 0;
}
