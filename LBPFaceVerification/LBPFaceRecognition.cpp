#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "LBPdescriptor.h"

using namespace cv::ml;

#define THRESH 100e3

void main() {
	Mat frame, gray, result;
	Rect faceRef;
	int descLen = histBinUni * colsBlk*colsBlk;
	float *descRef = (float*)calloc(descLen, sizeof(float));

	// Open video
	VideoCapture capture(0);
	if (!capture.isOpened())	return;
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Casecade classifier
	CascadeClassifier face_cascade;
	face_cascade.load("C:/opencv-3.4.1/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml");

	// Capture reference image
	while (1) {
		capture >> frame;
		result = frame.clone();

		// Find faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

		for (int i = 0; i < faces.size(); i++) {
			Point tl(faces[i].x, faces[i].y);
			Point br(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			rectangle(result, tl, br, Scalar(0, 255, 0), 3, 8, 0);
		}

		imshow("Reference", result);
		char ch = waitKey(10);
		if (ch == 32) {	// Space key
			// Save reference face region
			if (faces.size() == 1) {
				faceRef = Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
				destroyAllWindows();
				break;
			}
		}
	}

	// Calculate LBP desriptor of reference face image
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	getUniformLBPdescriptor(gray, faceRef, descRef);

	while (1) {
		capture >> frame;
		result = frame.clone();

		// Find faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

		// Convert RGB to grayscale
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// Draw faces
		for (int i = 0; i < faces.size(); i++) {
			// Compute LBP descriptor of target face
			float *descTar = (float*)calloc(descLen, sizeof(float));
			getUniformLBPdescriptor(gray, faces[i], descTar);

			// Compare similarity
			float dist = chiSquareDistance(descRef, descTar, descLen);

			// Draw box & Write score
			Point tl(faces[i].x, faces[i].y);
			Point br(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			if (dist < THRESH) {
				rectangle(result, tl, br, Scalar(0, 255, 0), 3, 8, 0);
				char str[64] = "";
				sprintf(str, "%.2fk", dist / 1000.0);
				putText(result, str, Point(tl.x, tl.y - 10), FONT_HERSHEY_SIMPLEX, 1., Scalar(0, 255, 0), 2);
			}
			else {
				rectangle(result, tl, br, Scalar(0, 0, 255), 3, 8, 0);
				char str[64] = "";
				sprintf(str, "%.2fk", dist / 1000.0);
				putText(result, str, Point(tl.x, tl.y - 10), FONT_HERSHEY_SIMPLEX, 1., Scalar(0, 0, 255), 2);
			}

			free(descTar);
		}

		// Display result
		imshow("Face verification", result);
		char ch = waitKey(10);
		if (ch == 27) {	// ESC key
			destroyAllWindows();
			break;
		}
	}

	free(descRef);
}