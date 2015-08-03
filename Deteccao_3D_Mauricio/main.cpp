#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <libfreenect_sync.h>
#include "kinect.hpp"
#include "detection.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
	int camera_id;
	uint32_t timestamp;
	uint16_t *depth_data, *buffer;
	vector<Vec4d> faces;

	if(argc > 1)
		camera_id = atoi(argv[1]);
	else
		camera_id = 0;

	// Initialize Kinect
	freenect_sync_get_depth((void **) &buffer, &timestamp, camera_id, FREENECT_DEPTH_11BIT);
	depth_data = new uint16_t[SIZE];
	Mat depth(HEIGHT, WIDTH, CV_16UC1, depth_data, WIDTH*sizeof(uint16_t));
	Mat vis(depth.size(), CV_8UC3);

	// Video loop
	for(;;) {
		// Capture new frame
		freenect_sync_get_depth((void **) &buffer, &timestamp, camera_id, FREENECT_DEPTH_11BIT);
		memcpy(depth_data, buffer, SIZE*sizeof(uint16_t));

		// BEGIN: Visualization
		for(int i=0; i < depth.rows; i++)
			for(int j=0; j < depth.cols; j++) {
				if(depth.at<uint16_t>(i,j) < 2047) {
					vis.at<Vec3b>(i,j)[0] = 0;
					if(depth.at<uint16_t>(i,j) < 1024) {
						vis.at<Vec3b>(i,j)[1] = 255-(depth.at<uint16_t>(i,j)/1023.0)*255.0;
						vis.at<Vec3b>(i,j)[2] = 255;
					}
					else {
						vis.at<Vec3b>(i,j)[1] = 0;
						vis.at<Vec3b>(i,j)[2] = ((depth.at<uint16_t>(i,j)-1024)/1023.0)*255.0;
					}
				}
				else {
					vis.at<Vec3b>(i,j)[0] = 0;
					vis.at<Vec3b>(i,j)[1] = 0;
					vis.at<Vec3b>(i,j)[2] = 0;
				}
			}
		// END: Visualization


		faces = face_detection(depth);
		// BEGIN: Visualization
		for(int i=0; i < faces.size(); i++)
			rectangle(vis, Point(faces[i][0]-faces[i][2],faces[i][1]-faces[i][2]), Point(faces[i][0]+faces[i][2],faces[i][1]+faces[i][2]), CV_RGB(0,255,0), 2, 8, 0);
		// END: Visualization

		imshow("3D Face Detection Demo", vis);

		if(waitKey(1) == 27)
			break;
	}

	freenect_sync_stop();

	return 0;
}

