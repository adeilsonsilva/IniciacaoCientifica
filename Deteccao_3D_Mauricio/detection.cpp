#include "detection.hpp"
#include "kinect.hpp"

void compute_projection(IplImage *p, IplImage *m, CvPoint3D64f *xyz, int n, double matrix[3][3], double background) {
	static int flag = 1, height, width, size, cx, cy, *li, *lj, *lc;
	int i, j, k, l, c, t;
	double d;

	if(flag) {
		flag = 0;

		height = p->height;
		width = p->width;
		size = height*width;

		li = (int *) malloc(3*size*sizeof(int));
		lj = li+size;
		lc = lj+size;

		cx = width/2;
		cy = height/2;
	}

	// Compute projection
	cvSet(p, cvRealScalar(-DBL_MAX), NULL);
	cvSet(m, cvRealScalar(0), NULL);
	for(i=0; i < n; i++) {
		j = cy-cvRound(xyz[i].x*matrix[1][0]+xyz[i].y*matrix[1][1]+xyz[i].z*matrix[1][2]);
		k = cx+cvRound(xyz[i].x*matrix[0][0]+xyz[i].y*matrix[0][1]+xyz[i].z*matrix[0][2]);
		d = xyz[i].x*matrix[2][0]+xyz[i].y*matrix[2][1]+xyz[i].z*matrix[2][2];

		if(j >= 0 && k >= 0 && j < height && k < width && d > CV_IMAGE_ELEM(p, double, j, k)) {
			CV_IMAGE_ELEM(p, double, j, k) = d;
			CV_IMAGE_ELEM(m, uchar, j, k) = 1;
		}
	}

	// Hole filling
	k=l=0;
	for(i=1; i < height-1; i++)
		for(j=1; j < width-1; j++)
			if(!CV_IMAGE_ELEM(m, uchar, i, j) && (CV_IMAGE_ELEM(m, uchar, i, j-1) || CV_IMAGE_ELEM(m, uchar, i, j+1) || CV_IMAGE_ELEM(m, uchar, i-1, j) || CV_IMAGE_ELEM(m, uchar, i+1, j))) {
				li[l] = i;
				lj[l] = j;
				lc[l] = 1;
				l++;
			}

	while(k < l) {
		i = li[k];
		j = lj[k];
		c = lc[k];
		if(!CV_IMAGE_ELEM(m, uchar, i, j) && i > 0 && i < height-1 && j > 0 && j < width-1 && c < FACE_HALF_SIZE) {
			CV_IMAGE_ELEM(m, uchar, i, j) = c+1;
			t = 0;
			d = 0.0f;
			if(CV_IMAGE_ELEM(m, uchar, i, j-1) && CV_IMAGE_ELEM(m, uchar, i, j-1) <= c) {
				t++;
				d += CV_IMAGE_ELEM(p, double, i, j-1);
			}
			else {
				li[l] = i;
				lj[l] = j-1;
				lc[l] = c+1;
				l++;
			}
			if(CV_IMAGE_ELEM(m, uchar, i, j+1) && CV_IMAGE_ELEM(m, uchar, i, j+1) <= c) {
				t++;
				d += CV_IMAGE_ELEM(p, double, i, j+1);
			}
			else {
				li[l] = i;
				lj[l] = j+1;
				lc[l] = c+1;
				l++;
			}
			if(CV_IMAGE_ELEM(m, uchar, i-1, j) && CV_IMAGE_ELEM(m, uchar, i-1, j) <= c) {
				t++;
				d += CV_IMAGE_ELEM(p, double, i-1, j);
			}
			else {
				li[l] = i-1;
				lj[l] = j;
				lc[l] = c+1;
				l++;
			}
			if(CV_IMAGE_ELEM(m, uchar, i+1, j) && CV_IMAGE_ELEM(m, uchar, i+1, j) <= c) {
				t++;
				d += CV_IMAGE_ELEM(p, double, i+1, j);
			}
			else {
				li[l] = i+1;
				lj[l] = j;
				lc[l] = c+1;
				l++;
			}
			CV_IMAGE_ELEM(p, double, i, j) = d/(double)t;
		}
		k++;
	}

	// Final adjustments
	for(i=0; i < height; i++)
		for(j=0; j < width; j++) {
			if(CV_IMAGE_ELEM(p, double, i, j) == -DBL_MAX)
				CV_IMAGE_ELEM(p, double, i, j) = background;
			if(CV_IMAGE_ELEM(m, uchar, i, j))
				CV_IMAGE_ELEM(m, uchar, i, j) = 1;
		}
}

// Compute rotation matrix and its inverse matrix
void computeRotationMatrix(double matrix[3][3], double imatrix[3][3], double aX, double aY, double aZ) {
	double cosX, cosY, cosZ, sinX, sinY, sinZ, d;

	cosX = cos(aX);
	cosY = cos(aY);
	cosZ = cos(aZ);
	sinX = sin(aX);
	sinY = sin(aY);
	sinZ = sin(aZ);

	matrix[0][0] = cosZ*cosY+sinZ*sinX*sinY;
	matrix[0][1] = sinZ*cosY-cosZ*sinX*sinY;
	matrix[0][2] = cosX*sinY;
	matrix[1][0] = -sinZ*cosX;
	matrix[1][1] = cosZ*cosX;
	matrix[1][2] = sinX;
	matrix[2][0] = sinZ*sinX*cosY-cosZ*sinY;
	matrix[2][1] = -cosZ*sinX*cosY-sinZ*sinY;
	matrix[2][2] = cosX*cosY;

	d = matrix[0][0]*(matrix[2][2]*matrix[1][1]-matrix[2][1]*matrix[1][2])-matrix[1][0]*(matrix[2][2]*matrix[0][1]-matrix[2][1]*matrix[0][2])+matrix[2][0]*(matrix[1][2]*matrix[0][1]-matrix[1][1]*matrix[0][2]);

	imatrix[0][0] = (matrix[2][2]*matrix[1][1]-matrix[2][1]*matrix[1][2])/d;
	imatrix[0][1] = -(matrix[2][2]*matrix[0][1]-matrix[2][1]*matrix[0][2])/d;
	imatrix[0][2] = (matrix[1][2]*matrix[0][1]-matrix[1][1]*matrix[0][2])/d;
	imatrix[1][0] = -(matrix[2][2]*matrix[1][0]-matrix[2][0]*matrix[1][2])/d;
	imatrix[1][1] = (matrix[2][2]*matrix[0][0]-matrix[2][0]*matrix[0][2])/d;
	imatrix[1][2] = -(matrix[1][2]*matrix[0][0]-matrix[1][0]*matrix[0][2])/d;
	imatrix[2][0] = (matrix[2][1]*matrix[1][0]-matrix[2][0]*matrix[1][1])/d;
	imatrix[2][1] = -(matrix[2][1]*matrix[0][0]-matrix[2][0]*matrix[0][1])/d;
	imatrix[2][2] = (matrix[1][1]*matrix[0][0]-matrix[1][0]*matrix[0][1])/d;
}

void xyz2depth(CvPoint3D64f *pt, double *i, double *j, double *s) {
	double z;

	z = DEPTH_Z4/RESOLUTION-pt->z;
	*i = (-pt->y/z)*DEPTH_FY+DEPTH_CY;
	*j = (pt->x/z)*DEPTH_FX+DEPTH_CX;
	*s = fabs(((pt->x+100.0)/z)*DEPTH_FX+DEPTH_CX-*j);
}

vector<Vec4d> face_detection_(Mat &depth, int minX, int maxX, int minY, int maxY, int minZ, int maxZ, double thr) {
	static int flag = 1, width, height, cx, cy;
	static IplImage *p, *v, *m, *sum, *sqsum, *tiltedsum, *msum, *sumint, *tiltedsumint;
	static CvPoint3D64f *xyz, *list, *clist;
	static CvPoint2D64f *xy;
	static double *z, background;
	static CvHaarClassifierCascade *face_cascade;
	int i, j, k, l, n, aX, aY, aZ;
	double matrix[3][3], imatrix[3][3], X, Y, Z;
	CvPoint3D64f avg;

	if(flag) {
		flag = 0;

		width = (int)(X_WIDTH*RESOLUTION);
		height = (int)(X_WIDTH*RESOLUTION);

		cx = width/2;
		cy = height/2;

		p = cvCreateImage(cvSize(width, height), IPL_DEPTH_64F, 1);
		m = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		v = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

		xyz = (CvPoint3D64f *) malloc(SIZE*sizeof(CvPoint3D64f));
		xy = (CvPoint2D64f *) malloc(SIZE*sizeof(CvPoint2D64f));
		z = (double *) malloc(DEPTH_RANGE*sizeof(double));

		for(i=0, k=0; i < HEIGHT; i++)
			for(j=0; j < WIDTH; j++, k++) {
				xy[k].x = (j-DEPTH_CX)/DEPTH_FX;
				xy[k].y = (i-DEPTH_CY)/DEPTH_FY;
			}

		for(i=0; i < DEPTH_RANGE; i++)
			z[i] = -DEPTH_Z3*tan(i/DEPTH_Z2+DEPTH_Z1)*RESOLUTION;
		background = z[DEPTH_THRESHOLD]+DEPTH_Z4;

		face_cascade = (CvHaarClassifierCascade *) cvLoad("ALL_Spring2003_3D.xml", 0, 0, 0);

		sum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_64F, 1);
		sqsum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_64F, 1);
		tiltedsum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_64F, 1);
		sumint = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_32S, 1);
		tiltedsumint = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_32S, 1);
		msum = cvCreateImage(cvSize(width+1, height+1), IPL_DEPTH_32S, 1);

		list = (CvPoint3D64f *) malloc(2000*sizeof(CvPoint3D64f));
		clist = list+1000;
	}

	// Convert depth to 3D coordinates with grid sampling
	for(i=0, k=0, n=0; i < HEIGHT; i+=6, k=i*WIDTH)
		for(j=0; j < WIDTH; j+=6, k+=6) {
			l = depth.at<uint16_t>(i,j);
			if(l < thr) {
				xyz[n].x = -xy[k].x*z[l];
				xyz[n].y = xy[k].y*z[l];
				xyz[n].z = z[l]+DEPTH_Z4;
				n++;
			}
		}

	// Detection loop
	for(aX=minX, k=0; aX <= maxX; aX += 10) {
	for(aY=minY; aY <= maxY; aY += 10) {
	for(aZ=minZ; aZ <= maxZ; aZ += 10) {
		if(aX+aY+aZ > 30)
			continue;

		//aY = 0;
		//aZ = 0;

		computeRotationMatrix(matrix, imatrix, aX*0.017453293, aY*0.017453293, aZ*0.017453293);
		compute_projection(p, m, xyz, n, matrix, background);

		cvIntegral(p, sum, sqsum, tiltedsum);
		cvIntegral(m, msum, NULL, NULL);

		for(i=0; i < height+1; i++)
			for(j=0; j < width+1; j++) {
				CV_IMAGE_ELEM(sumint, int, i, j) = CV_IMAGE_ELEM(sum, double, i, j);
				CV_IMAGE_ELEM(tiltedsumint, int, i, j) = CV_IMAGE_ELEM(tiltedsum, double, i, j);
			}

		cvSetImagesForHaarClassifierCascade(face_cascade, sumint, sqsum, tiltedsumint, 1.0);

		for(i=0; i < height-20; i++)
			for(j=0; j < width-20; j++)
				if(CV_IMAGE_ELEM(msum, int, i+FACE_SIZE, j+FACE_SIZE)-CV_IMAGE_ELEM(msum, int, i, j+FACE_SIZE)-CV_IMAGE_ELEM(msum, int, i+FACE_SIZE, j)+CV_IMAGE_ELEM(msum, int, i, j) == 441)
					if(cvRunHaarClassifierCascade(face_cascade, cvPoint(j,i), 0) > 0) {
						X = (j+FACE_HALF_SIZE-cx)/RESOLUTION;
						Y = (cy-i-FACE_HALF_SIZE)/RESOLUTION;
						Z = (CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE+6, j+FACE_HALF_SIZE+6)-CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE-5, j+FACE_HALF_SIZE+6)-CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE+6, j+FACE_HALF_SIZE-5)+CV_IMAGE_ELEM(sum, double, i+FACE_HALF_SIZE-5, j+FACE_HALF_SIZE-5))/121.0/RESOLUTION;

						list[k].x = X*imatrix[0][0]+Y*imatrix[0][1]+Z*imatrix[0][2];
						list[k].y = X*imatrix[1][0]+Y*imatrix[1][1]+Z*imatrix[1][2];
						list[k].z = X*imatrix[2][0]+Y*imatrix[2][1]+Z*imatrix[2][2];
						k++;
					}
	}
	}
	}

	// Merge multiple detections
	vector<Vec4d> r;
	Vec4d tmp;
	while(k > 0) {
		avg.x = clist[0].x = list[0].x;
		avg.y = clist[0].y = list[0].y;
		avg.z = clist[0].z = list[0].z;
		list[0].x = DBL_MAX;

		j=1;
		for(l=0; l < j; l++)
			for(i=1; i < k; i++)
				if(list[i].x != DBL_MAX) {
					X = sqrt(pow(list[i].x-clist[l].x, 2.0)+pow(list[i].y-clist[l].y, 2.0)+pow(list[i].z-clist[l].z, 2.0));
					if(X < 50.0) {
						avg.x += clist[j].x = list[i].x;
						avg.y += clist[j].y = list[i].y;
						avg.z += clist[j].z = list[i].z;
						list[i].x = DBL_MAX;
						j++;
					}
				}

		avg.x /= j;
		avg.y /= j;
		avg.z /= j;

		xyz2depth(&avg, &tmp[1], &tmp[0], &tmp[2]);
		tmp[3] = j;
		r.push_back(tmp);

		j=0;
		for(i=1; i < k; i++)
			if(list[i].x != DBL_MAX) {
				list[j].x = list[i].x;
				list[j].y = list[i].y;
				list[j].z = list[i].z;
				j++;
			}
		k=j;
	}

	return r;
}

vector<Vec4d> face_detection(Mat &depth) {
	return face_detection_(depth, 0, 30, -20, 20, 0, 0, DEPTH_THRESHOLD);
}

vector<Vec4d> frontal_face_detection(Mat &depth) {
	return face_detection_(depth, 0, 0, 0, 0, 0, 0, DEPTH_CTHRESHOLD);
}

