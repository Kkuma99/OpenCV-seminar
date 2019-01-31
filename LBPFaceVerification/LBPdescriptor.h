#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

#define PI 3.1415926535
#define modelSize 128
#define blkSize 16
#define colsBlk (modelSize / (blkSize / 2) - 1)	// # of blks in x axis
#define histBin 256
#define histBinUni 59	// Histogram bin for uniform LBP

static int uniform[256] = {
	0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
	14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
	58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
	58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
	58,58,58,50,51,52,58,53,54,55,56,57
};

void getLBPdescriptor(Mat img, Rect roi, float *desc);
void getUniformLBPdescriptor(Mat img, Rect roi, float *desc);
void makeLBPimage(Mat img, Mat &LBPimg);
void makeCircularLBPimage(Mat img, Mat &LBPimg, int p, int r);
float EuclideanDistance(float *descRef, float *descTar, int len);
float chiSquareDistance(float *descRef, float *descTar, int len);