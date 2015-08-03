
// Image parameters
#define WIDTH 640							// Input image width
#define HEIGHT 480							// Input image height
#define SIZE 307200							// Input image size

// Calibration parameters
#define DEPTH_RANGE 2048					// Range of depth values [0,DEPTH_RANGE[
#define DEPTH_FX 5.8498272251689014e+02		// Scaling factor for the x axis
#define DEPTH_FY 5.8509835924680374e+02		// Scaling factor for the y axis
#define DEPTH_CX 3.1252165122981484e+02		// Camera center for the x axis
#define DEPTH_CY 2.3821622578866226e+02		// Camera center for the y axis
#define DEPTH_Z1 1.1863						// Disparity to mm - 1st parameter
#define DEPTH_Z2 2842.5						// Disparity to mm - 2nd parameter
#define DEPTH_Z3 123.6						// Disparity to mm - 3rd parameter
#define DEPTH_Z4 95.45454545				// Disparity to mm - 4th parameter (750*RESOLUTION)
#define DEPTH_LTHRESHOLD 400				// Minimum disparity value
#define DEPTH_CTHRESHOLD 675					// Maximum disparity value
#define DEPTH_THRESHOLD 875					// Maximum disparity value

// Detection parameters
#define X_WIDTH 1800.0						// Orthogonal projection width - in mm
#define Y_WIDTH 1600.0						// Orthogonal projection height - in mm
#define RESOLUTION 0.127272727				// Resolution in pixels per mm
#define FACE_SIZE 21						// Face size - 165*RESOLUTION
#define FACE_HALF_SIZE 10					// (165*RESOLUTION)/2

// Normalization parameters
#define MODEL_WIDTH 48.0
#define MODEL_HEIGHT_1 56.0
#define MODEL_HEIGHT_2 16.0
#define MODEL_RESOLUTION 1.0
#define MAX_DEPTH_VALUE 5000
#define OUTLIER_THRESHOLD 15.0
#define OUTLIER_SQUARED_THRESHOLD 225.0
#define MAX_ICP_ITERATIONS 200

