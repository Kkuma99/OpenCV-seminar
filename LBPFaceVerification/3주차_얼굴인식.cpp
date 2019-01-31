#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

#define SIZE 128
#define BLKSIZE 16
#define THRESH 700

void LBP(Mat input, Mat output);
void make_descriptor(Mat output, float *descriptor);

void main() {
	Mat frame, result;
	Rect faceRegionRef;
	Mat faceImgRef;
	Rect faceRegionTarget;
	Mat faceImgTarget;
	float *descriptorRef = (float *)calloc(256 * 15 * 15, sizeof(float));
	float *descriptorTarget = (float *)calloc(256 * 15 * 15, sizeof(float));

	// Open video
	VideoCapture capture(0);
	if (!capture.isOpened())   return;
	capture.set(CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CAP_PROP_FRAME_HEIGHT, 480);

	// Casecade classifier
	CascadeClassifier face_cascade;
	face_cascade.load("C:/opencv4.0.0/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml");

	// Capture reference image
	while (1) {
		capture >> frame;
		result = frame.clone();

		// Find faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | 2, Size(10, 10));

		for (int i = 0; i < faces.size(); i++) {
			Point tl(faces[i].x, faces[i].y);
			Point br(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			rectangle(result, tl, br, Scalar(0, 255, 0), 3, 8, 0);
		}

		imshow("Reference", result);
		char ch = waitKey(10);
		if (ch == 32) {   // Space key
		// Save reference face region
			if (faces.size() == 1) {
				faceRegionRef = Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
				destroyAllWindows();
				break;
			}
		}
	}

	// Save face
	faceImgRef = frame(faceRegionRef).clone();
	imwrite("face_ref.jpg", faceImgRef);

	int height = faceImgRef.rows;
	int width = faceImgRef.cols;

	// 흑백으로 변환
	Mat grayimg(height, width, CV_8UC1);
	cvtColor(faceImgRef, grayimg, COLOR_BGR2GRAY);

	// 128*128로 리사이즈
	resize(grayimg, grayimg, Size(SIZE, SIZE));

	// LBP이미지 구하기
	Mat LBP_img(SIZE, SIZE, CV_8UC1);
	LBP(faceImgRef, LBP_img);

	// descriptor 구하기
	make_descriptor(LBP_img, descriptorRef);

	// -------------------------------------------

	// Capture target image
	while (1) {
		capture >> frame;
		result = frame.clone();

		// Find faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | 2, Size(10, 10));

		if (faces.size() >= 1) {
			// 얼굴 이미지 잘라오기
			faceRegionTarget = Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
			faceImgTarget = frame(faceRegionTarget).clone();

			int height = faceImgTarget.rows;
			int width = faceImgTarget.cols;

			// 흑백으로 변환
			Mat grayimg(height, width, CV_8UC1);
			cvtColor(faceImgTarget, grayimg, COLOR_BGR2GRAY);

			// 128*128로 리사이즈
			resize(grayimg, grayimg, Size(SIZE, SIZE));

			// LBP이미지 구하기
			Mat LBP_img(SIZE, SIZE, CV_8UC1);
			LBP(faceImgTarget, LBP_img);

			// descriptor 구하기
			make_descriptor(LBP_img, descriptorTarget);

			// 거리 구하기
			float distance = 0;
			for (int i = 0; i < 256 * 15 * 15; i++) {
	
				distance += fabs(descriptorRef[i] - descriptorTarget[i]);
	
			}
			printf("%f\n", distance);

			// 얼굴 주변에 네모 그리기
			Point tl(faces[0].x, faces[0].y);
			Point br(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
			if (distance < THRESH) {
				rectangle(result, tl, br, Scalar(0, 255, 0), 3, 8, 0);
			}
			else {
				rectangle(result, tl, br, Scalar(0, 0, 255), 3, 8, 0);
			}

		}

		imshow("Face Verification", result);
		char ch = waitKey(10);
		if (ch == 27)	break;	// ESC key
	}


	free(descriptorRef);
	free(descriptorTarget);

}

void LBP(Mat input, Mat output)
{
	int height = input.rows;
	int width = input.cols;

	int coord[8][2] = { {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1} };

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			int value = 0;

			for (int i = 0; i < 8; i++) {

				int xx = x + coord[i][0];
				int yy = y + coord[i][1];

				if ((xx >= 0 && xx < width) && (yy >= 0 && yy < height)) {

					if ((input.at<uchar>(y, x)) > (input.at<uchar>(yy, xx)))
						value += 0 * pow(2, 7 - i);
					else
						value += 1 * pow(2, 7 - i);

				}
			}

			output.at<uchar>(y, x) = value;

		}
	}
}

void make_descriptor(Mat output, float * descriptor)
{
	int height = output.rows;
	int width = output.cols;

	// 128*128 이미지에 대한 Histogram
	for (int y = 0; y < height; y += BLKSIZE / 2) {
		for (int x = 0; x < width; x += BLKSIZE / 2) {

			float *histogram = (float *)calloc(256, sizeof(float));
			for (int yy = y; yy < y + BLKSIZE; yy++) {
				for (int xx = x; xx < x + BLKSIZE; xx++) {
					int value = output.at<uchar>(yy, xx);
					histogram[value]++;
				}
			}
			// normalize
			float denom = 0;
			for (int i = 0; i < 256; i++)
				denom += pow(histogram[i], 2);
			denom = sqrt(denom);

			for (int i = 0; i < 256; i++) {
				int index = 256 * ((y / 8) * 15 + (x / 8)) + i;
				descriptor[index] = histogram[i] / denom;
			}

			free(histogram);

		}
	}
}
