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

const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.45;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;     //0.80
const double DESIRED_LEFT_EYE_X = 0.26;     // Controls how much of the face is visible after preprocessing. 0.16 and 0.14
const double DESIRED_LEFT_EYE_Y = 0.24;

using namespace cv;
using namespace std;

bool protonect_shutdown = false;

void sigint_handler(int s)
{
  protonect_shutdown = true;
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

    int frames_saved = 0;
    cout << getTickFrequency() << endl;
    while(!protonect_shutdown)
    {
    	listener.waitForNewFrame(frames);
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        Mat frame = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data)/ 80000;
        frame.convertTo(frame, CV_8UC3, 255, 0);
        frames_saved++;
        cv::imwrite(format("/home/matheusm/Record/frame%d.pgm", frames_saved), frame);
        string mesage = format("Imagem salva numero %d", frames_saved);
        putText(frame, mesage, Point(8, frame.rows - 8), FONT_HERSHEY_PLAIN, 1.2, CV_RGB(255,255,255), 1.0);
        cv::imshow("Record Video", frame);
        int key = cv::waitKey(1);
        protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape
        if(frames_saved > 1004)
            protonect_shutdown = true;
        listener.release(frames);
    }

    dev->stop();
    dev->close();

    return 0;
}