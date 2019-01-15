#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define file_store

#define PI 3.1415926535
#define HISTBIN 9

int main() {
	// Image read
	Mat imgColor = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	int height = imgColor.rows;
	int width = imgColor.cols;
	Mat imgGray(height, width, CV_8UC1);
	Mat imgEdge(height, width, CV_8UC1);

	float *magnitude = (float *)calloc(height*width, sizeof(float));
	float *phase = (float *)calloc(height*width, sizeof(float));
	float *hist = (float *)calloc(HISTBIN, sizeof(float));
	int kernelx[3][3] = {
		{ -1,0,1 },
		{ -1,0,1 },
		{ -1,0,1 }
	};
	int kernely[3][3] = {
		{ -1,-1,-1 },
		{ 0,0,0 }, 
		{ 1,1,1 }
	};

	// RGB to Gray
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uchar b = imgColor.at<Vec3b>(y, x)[0];
			uchar g = imgColor.at<Vec3b>(y, x)[1];
			uchar r = imgColor.at<Vec3b>(y, x)[2];
			uchar gray = (b + g + r) / 3;
			imgGray.at<uchar>(y, x) = gray;
		}
	}

	// Get edge information
	float max = -1;
	float min = 100000;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Set range
			// yy : -1 ~ 1, xx : -1 ~ 1
			int start_x = (x == 0) ? 0 : -1;		// starting index (x axis)
			int end_x = (x == width - 1) ? 0 : 1;	// ending index (x axis)
			int start_y = (y == 0) ? 0 : -1;		// starting index (y axis)
			int end_y = (y == height - 1) ? 0 : 1;	// ending index (y axis)

			float fx = 0;
			float fy = 0;
			for (int yy = start_y; yy <= end_y; yy++) {
				for (int xx = start_x; xx <= end_x; xx++) {
					fx += kernelx[yy + 1][xx + 1] * imgGray.at<uchar>(y + yy, x + xx);
					fy += kernely[yy + 1][xx + 1] * imgGray.at<uchar>(y + yy, x + xx);
				}
			}
			// Calculate magnitude
			float magVal = sqrt(fx*fx + fy * fy);

			// Get min/max for normalization
			min = (magVal < min) ? magVal : min;
			max = (magVal > max) ? magVal : max;

			magnitude[y*width + x] = magVal;

			// Caculate phase (0 ~ 180)
			float phaseVal = atan2(fy, fx) * 180 / PI;
			if (phaseVal < 0)		phaseVal += 180;
			if (phaseVal >= 180)	phaseVal = 0;

			phase[y*width + x] = phaseVal;
		}
	}

	// Normalize magnitude
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float val = (magnitude[y*width + x] - min) / (max - min) * 255;
			imgEdge.at<uchar>(y, x) = (uchar)val;
		}
	}

	// Get histogram
	int interval = 180 / HISTBIN;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int idx = phase[y*width + x] / interval;
			hist[idx]++;
		}
	}
	
#ifdef file_store
	// Save histogram as excel file
	FILE *fp;
	fp = fopen("histogram.csv", "w");
	for (int i = 0; i < 9; i++) {
		fprintf(fp, "%f\n", hist[i]);
	}
	fclose(fp);
#endif

	// Display results
	imshow("original", imgColor);
	imshow("gray", imgGray);
	imshow("edge", imgEdge);
	waitKey(0);

	destroyAllWindows();
	free(magnitude);
	free(phase);
	free(hist);

	return 0;
}