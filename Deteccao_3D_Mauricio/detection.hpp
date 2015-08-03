#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

vector<Vec4d> face_detection(Mat &depth);
vector<Vec4d> frontal_face_detection(Mat &depth);

