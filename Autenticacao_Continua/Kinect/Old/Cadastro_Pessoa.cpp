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
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

using namespace cv;
using namespace std;

const string PATH_CASCADE_FACE = "/home/matheusm/libfreenect2/examples/protonect/Cascades/cascade.xml";

bool protonect_shutdown = false;
Rect m_rcBtnAdd;
bool save = false;

void sigint_handler(int s)
{
  protonect_shutdown = true;
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
  int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner

  Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
  Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

  string lsLeftEye_haar = "/home/matheusm/Cascades/haarcascade_mcs_lefteye_alt.xml";
  string lsRightEye_haar = "/home/matheusm/Cascades/haarcascade_mcs_righteye_alt.xml";
  string lsBothEyes_haar = "/home/matheusm/Cascades/haarcascade_eye.xml";

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
    //rectangle(face, eye_i, CV_RGB(255, 255, 255), 1);
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
      //rectangle(face, eye_i, CV_RGB(255, 255, 255), 1);
    }
  }
  haar_cascade.load(lsRightEye_haar);
  haar_cascade.detectMultiScale(topRightOfFace, detectedRightEye);
  for(int i = 0; i < detectedRightEye.size(); i++) {
    Rect eye_i = detectedRightEye[i];
    eye_i.x += rightX;;
    eye_i.y += topY;
    rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
    //rectangle(face, eye_i, CV_RGB(255, 255, 255), 1);
  }
  if(detectedRightEye.empty()) {
    haar_cascade.load(lsBothEyes_haar);
    haar_cascade.detectMultiScale(topLeftOfFace, detectedRightEye);
    for(int i = 0; i < detectedRightEye.size(); i++) {
      Rect eye_i = detectedRightEye[i];
      eye_i.x += leftX;
      eye_i.y += topY;
      rightEye = Point(eye_i.x + eye_i.width/2, eye_i.y + eye_i.height/2);
      //rectangle(face, eye_i, CV_RGB(255, 255, 255), 1);
    }
  }
  //if both eyes were detected
  Mat warped;
  if (leftEye.x >= 0 && rightEye.x >= 0 && ((rightEye.x - leftEye.x) > (face.cols/4) )) {
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
    sucess = true;
  }
  else {
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

// Draw text into an image. Defaults to top-left-justified text, but you can give negative x coords for right-justified text,
// and/or negative y coords for bottom-justified text.
// Returns the bounding rect around the drawn text.
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    // Get the text size & baseline.
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Adjust the coords for left/right-justified or top/bottom-justified.
    if (coord.y >= 0) {
        // Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
        coord.y += textSize.height;
    }
    else {
        // Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
        coord.y += img.rows - baseline + 1;
    }
    // Become right-justified if desired.
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }

    // Get the bounding box around the text.
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Draw anti-aliased text.
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    // Let the user know how big their text is, in case they want to arrange things.
    return boundingRect;
}

// Draw a GUI button into the image, using drawString().
// Can specify a minWidth if you want several buttons to all have the same width.
// Returns the bounding rect around the drawn button, allowing you to position buttons next to each other.
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Get the bounding box around the text.
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // Draw a filled rectangle around the text.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);

    // Draw the actual text that will be displayed, using anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(10,55,20));

    return rcButton;
}

bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

// Mouse event handler. Called automatically by OpenCV when the user clicks in the GUI window.
void onMouse(int event, int x, int y, int, void*)
{
    // We only care about left-mouse clicks, not right-mouse clicks or mouse movement.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;
    // Check if the user clicked on one of our GUI buttons.
    Point pt = Point(x,y);
    if (isPointInRect(pt, m_rcBtnAdd)) {
        save = true;
    }
}

int main(int argc, char *argv[])
{
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

  CascadeClassifier haar_cascade;
  haar_cascade.load(PATH_CASCADE_FACE);

  int numFaces = 1;
  int frame_count = 0;
  // Create a GUI window for display on the screen.
  namedWindow("Cadastrar Pessoa"); // Resizable window, might not work on Windows.
  // Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
  setMouseCallback("Cadastrar Pessoa", onMouse, 0);

  while(!protonect_shutdown)
  {
    int key = cv::waitKey(1);
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    Mat frame = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data)/ 80000;
    frame.convertTo(frame, CV_8UC3, 255, 0);
    vector< Rect_<int> > faces;
    cout << frame_count++ << endl;
    haar_cascade.detectMultiScale(frame, faces);

    m_rcBtnAdd = drawButton(frame, "Add Person", Point(BORDER, BORDER));
    bool sucess = false;

    if(faces.size() != 0) {
        Mat face = frame(faces[0]);
        Mat faceNormalized = faceNormalize(face, 200, sucess);
        Mat srcBGR = faceNormalized;
        cv::resize(srcBGR, srcBGR, Size(100, 100), 1.0, 1.0, INTER_CUBIC);
        int cx = (frame.cols - srcBGR.cols) - BORDER;
        Rect dstRC = Rect(cx, BORDER, srcBGR.cols, srcBGR.cols);
        Mat dstROI = frame(dstRC);
        // Copy the pixels from src to dst.
        srcBGR.copyTo(dstROI);
        if(save && sucess) {
          cv::imwrite(format("/home/matheusm/StremensDataBase/image%d.pgm", numFaces), faceNormalized);
          //save = false;
          numFaces++;
          string mesage = format("Imagem salva numero %d", numFaces-1);
          putText(frame, mesage, Point(BORDER, frame.rows - BORDER), FONT_HERSHEY_PLAIN, 1.7, CV_RGB(255,255,255), 1.0);
        }
    }
    for(int i = 0; i < faces.size(); i++) {
      Rect face_i = faces[i];
      rectangle(frame, face_i, CV_RGB(0, 0, 255), 1);
    }
    //cv::resize(frame, frame, Size(frame.cols*1.5, frame.rows*1.5), 1.0, 1.0, INTER_CUBIC);
    cv::imshow("Cadastrar Pessoa", frame);
    protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape
    if(numFaces > 100)
      protonect_shutdown = true;
   
    listener.release(frames);
  }

  dev->stop();
  dev->close();

  return 0;
}
