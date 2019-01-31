#include "LBPdescriptor.h"

// img : grayscale
void getLBPdescriptor(Mat img, Rect roi, float *desc) {
	// Resize given image
	Mat roi_img = img(roi).clone();
	Mat model;
	resize(roi_img, model, Size(modelSize, modelSize));

	// Memory allocation for LBP image
	Mat LBPimg = Mat::zeros(modelSize, modelSize, CV_8UC1);

	// Make LBP image of given model
	makeLBPimage(model, LBPimg);

	// Get LBP desriptor of given roi
	// (xb, yb) : block index
	for (int yb = 0; yb < colsBlk; yb++) {
		for (int xb = 0; xb < colsBlk; xb++) {
			// Memory allocation for block histogram
			float *hist = (float*)calloc(histBin, sizeof(float));

			// Caculate histogram of current block
			// (x, y) : coordinate in block
			for (int y = 0; y < blkSize; y++) {
				for (int x = 0; x < blkSize; x++) {
					// (xx, yy) : coordinate in model image
					int yy = yb * (blkSize / 2) + y;
					int xx = xb * (blkSize / 2) + x;

					// Find histogram index and stack corresponding edge magnitude
					hist[LBPimg.at<uchar>(yy, xx)]++;
				}
			}

			// L2 normalization
			float norm = 0;	// normalization factor
			for (int i = 0; i < histBin; i++) {
				norm += hist[i] * hist[i];
			}
			norm = sqrt(norm);
			for (int i = 0; i < histBin; i++) {
				int idx = (yb*colsBlk + xb)*histBin + i;	// descriptor index
				desc[idx] = hist[i] / norm;
			}

			free(hist);
		}
	}
}

void getUniformLBPdescriptor(Mat img, Rect roi, float *desc) {
	// Resize given image
	Mat roi_img = img(roi).clone();
	Mat model;
	resize(roi_img, model, Size(modelSize, modelSize));

	// Memory allocation for LBP image
	Mat LBPimg = Mat::zeros(modelSize, modelSize, CV_8UC1);

	// Make LBP image of given model
	makeLBPimage(model, LBPimg);

	// Get LBP desriptor of given roi
	// (xb, yb) : block index
	for (int yb = 0; yb < colsBlk; yb++) {
		for (int xb = 0; xb < colsBlk; xb++) {
			// Memory allocation for block histogram
			float *hist = (float*)calloc(histBinUni, sizeof(float));

			// Caculate histogram of current block
			// (x, y) : coordinate in block
			for (int y = 0; y < blkSize; y++) {
				for (int x = 0; x < blkSize; x++) {
					// (xx, yy) : coordinate in model image
					int yy = yb * (blkSize / 2) + y;
					int xx = xb * (blkSize / 2) + x;

					// Find histogram index and stack corresponding edge magnitude
					int idx = uniform[LBPimg.at<uchar>(yy, xx)];
					hist[idx]++;
				}
			}

			// L2 normalization
			float norm = 0;	// normalization factor
			for (int i = 0; i < histBinUni; i++) {
				norm += hist[i] * hist[i];
			}
			norm = sqrt(norm);
			for (int i = 0; i < histBinUni; i++) {
				int idx = (yb*colsBlk + xb)*histBinUni + i;	// descriptor index
				desc[idx] = hist[i] / norm;
			}

			free(hist);
		}
	}
}

void makeLBPimage(Mat img, Mat &LBPimg) {
	int height = img.rows;
	int width = img.cols;
	// Clockwise coordinate from current pixel (x, y)
	int coord[8][2] = {	
	{0, -1}, {1, -1}, {1, 0}, {1, 1},
	{0, 1}, {-1, 1}, {-1, 0}, {-1, -1}
	};

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uchar LBP = 0;
			for (int i = 0; i < 8; i++) {
				int xx = x + coord[i][0];
				int yy = y + coord[i][1];
				if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
					LBP += pow(2, i)*((img.at<uchar>(y, x) > img.at<uchar>(yy, xx)) ? 1 : 0);
				}
			}
			LBPimg.at<uchar>(y, x) = LBP;
		}
	}
}

void makeCircularLBPimage(Mat img, Mat &LBPimg, int p, int r) {
	int height = img.rows;
	int width = img.cols;
	float angle = 2 * PI / p;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uchar LBP = 0;
			for (int i = 0; i < p; i++) {
				float xx = x + (float)r * cos(i*angle);
				float yy = y + (float)r * sin(i*angle);
				int x0 = (int)xx;	// x0 pos
				int y0 = (int)y0;	// y0 pos
				float wx = xx - x0;	// weight in x axis
				float wy = yy - y0;	// weight in y axis
				int x1 = (x0 < width - 1) ? x0 + 1 : x0;	// condition : x1 < width
				int y1 = (y0 < height - 1) ? y0 + 1 : y0;	// condition : y1 < height

				// Interpolation
				float uVal, dVal, lVal, rVal = 0;
				if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
					float uVal = (1 - wx)*img.at<uchar>(y0, x0) + wx * img.at<uchar>(y0, x1);
					float dVal = (1 - wx)*img.at<uchar>(y1, x0) + wx * img.at<uchar>(y1, x1);
					float lVal = (1 - wy)*img.at<uchar>(y0, x0) + wy * img.at<uchar>(y1, x0);
					float rVal = (1 - wy)*img.at<uchar>(y0, x1) + wy * img.at<uchar>(y1, x1);
				}
				uchar Val = ((1 - wx)*lVal + wx * rVal + (1 - wy)*uVal + wy * dVal) / 2;

				LBP += pow(2, i)*((img.at<uchar>(y, x) > Val) ? 1 : 0);
			}
			LBPimg.at<uchar>(y, x) = LBP;
		}
	}
}

float EuclideanDistance(float *descRef, float *descTar, int len) {
	float dist = 0;

	for (int i = 0; i < len; i++) {
		dist += fabs(descRef[i] - descTar[i]);
	}

	return dist;
}

float chiSquareDistance(float *descRef, float *descTar, int len) {
	float delta = 0.0001;	// For zero-division
	float dist = 0;

	for (int i = 0; i < len; i++) {
		dist += pow(descRef[i] - descTar[i], 2) / (descRef[i] + delta);
	}

	return dist;
}