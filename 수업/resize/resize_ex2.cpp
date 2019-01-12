#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	// Read image
	Mat imgOrig = imread("lena.jpg", 0);
	int hOrig = imgOrig.rows;
	int wOrig = imgOrig.cols;
	cout << "Original image height : " << hOrig << endl;
	cout << "Original image width : " << wOrig << endl;

	// Get scale factor
	float scale;
	cout << "Input scale factor : ";
	cin >> scale;
	int hResult = imgOrig.rows*scale;
	int wResult = imgOrig.cols*scale;
	Mat imgResult = Mat::zeros(hResult, wResult, CV_8UC1);
	cout << "Result image height : " << hResult << endl;
	cout << "Result image width : " << wResult << endl;

	for (int y = 0; y < hResult; y++) {
		for (int x = 0; x < wResult; x++) {
			// (y0, x0)       ^     (y0, x1)
			//                |
			//               wy
			//                |
			// <-----wx----->(-)
			//
			// (y1, x0)             (y1, x1)

			int x0 = x / scale;	// x0 pos at original image
			int y0 = y / scale;	// y0 pos at original image
			float wx = x / scale - x0;	// weight in x axis
			float wy = y / scale - y0;	// weight in y axis

			int x1 = (x0 < wOrig - 1) ? x0 + 1 : x0;	// condition : x1 < width
			int y1 = (y0 < hOrig - 1) ? y0 + 1 : y0;	// condition : y1 < height

			// Get pixel values
			uchar p1 = imgOrig.at<uchar>(y0, x0);
			uchar p2 = imgOrig.at<uchar>(y0, x1);
			uchar p3 = imgOrig.at<uchar>(y1, x0);
			uchar p4 = imgOrig.at<uchar>(y1, x1);
			// Calculate new value
			float tVal = (1 - wx)*p1 + wx * p2;
			float bVal = (1 - wx)*p3 + wx * p4;
			float lVal = (1 - wy)*p1 + wy * p3;
			float rVal = (1 - wy)*p2 + wy * p4;
			uchar Val = ((1 - wx)*lVal + wx * rVal + (1 - wy)*tVal + wy * bVal) / 2;
			imgResult.at<uchar>(y, x) = Val;
		}
	}

	// Show results
	imwrite("img_resize.bmp", imgResult);
	imshow("original", imgOrig);
	imshow("scaled", imgResult);
	waitKey(0);

	return 0;
}