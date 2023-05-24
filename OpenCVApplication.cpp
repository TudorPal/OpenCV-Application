// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#define NOMINMAX
#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>
#include <cmath>

wchar_t* projectPath;

std::default_random_engine gen;
std::uniform_int_distribution<int> d(0, 255);

int factor;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the �diblook style�
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void additiveGray() //daca trece de 255, te opresti la 255
{
	int factor = 0;
	std::cin >> factor;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE); //citeste imaginea in grayscale

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel + factor <= 255)
					dst.at<uchar>(i, j) = pixel + factor;
				else
					dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void multiplicativeGray()
{
	int factor = 0;
	std::cin >> factor;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE); //citeste imaginea in grayscale

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel * factor <= 255)
					dst.at<uchar>(i, j) = pixel * factor;
				else
					dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void creareImagine4col() {

	int height = 256;
	int width = 256;
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i < 128 && j < 128)
			{
				img.at<Vec3b>(i, j)[0] = 255;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 255;
			}
			if (i < 128 && j >= 128)
			{
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 255;
			}
			if (i >= 128 && j < 128)
			{
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 0;
			}
			if (i >= 128 && j >= 128)
			{
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	imshow("imagine", img);
	waitKey();
}

void colorChannels()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, 1); //citeste imaginea in grayscale

		int height = src.rows;
		int width = src.cols;

		Mat red = Mat(height, width, CV_8UC1);
		Mat green = Mat(height, width, CV_8UC1);
		Mat blue = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				blue.at<uchar>(i, j) = v3[0];
				green.at<uchar>(i, j) = v3[1];
				red.at<uchar>(i, j) = v3[2];
			}
		}

		imshow("input image", src);
		imshow("red gray", red);
		imshow("green gray", green);
		imshow("blue gray", blue);
		waitKey();
	}
}

void blackWhite()
{
	int factor = 0;
	std::cout << "prag = ";
	std::cin >> factor;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE); //citeste imaginea in grayscale

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel < factor)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("gray image", src);
		imshow("black-white image", dst);
		waitKey();
	}
}

void testRGB2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, IMREAD_COLOR); //citeste imaginea color

		int height = src.rows;
		int width = src.cols;

		Mat H_norm = Mat(height, width, CV_8UC1);
		Mat S_norm = Mat(height, width, CV_8UC1);
		Mat V_norm = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float H, S, V;
				Vec3b v3 = src.at<Vec3b>(i, j);
				float b = (float)v3[0] / 255;
				float g = (float)v3[1] / 255;
				float r = (float)v3[2] / 255;

				float M = max(r, max(g, b));
				float m = min(r, min(g, b));
				float C = M - m;

				//Value
				V = M;

				//Saturation
				if (V == 0)
					S = 0;
				else
					S = C / V;

				//Hue
				if (C == 0)
					H = 0;
				else {
					if (M == r)
						H = 60 * ((g - b) / C);
					if (M == g)
						H = 120 + 60 * (b - r) / C;
					if (M == b)
						H = 240 + 60 * (r - g) / C;
				}
				if (H < 0)
					H += 360;
				H_norm.at<uchar>(i, j) = H * 255 / 360;
				S_norm.at<uchar>(i, j) = S * 255;
				V_norm.at<uchar>(i, j) = V * 255;
			}
		}

		imshow("input image", src);
		imshow("H gray", H_norm);
		imshow("S gray", S_norm);
		imshow("V gray", V_norm);

		waitKey();
	}
}

void isInside2(int i, int j) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, 1); //citeste imaginea in grayscale

		int height = src.rows;
		int width = src.cols;

		if (i <= height && j <= width && i >= 0 && j >= 0)
			printf("true\n");
		else
			printf("false\n");
	}
}

bool isInside(Mat src, int i, int j) {
	int height = src.rows;
	int width = src.cols;

	if (i <= height && j <= width && i >= 0 && j >= 0)
		return true;
	return false;
}

int* testHistogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE); //citeste imaginea in grayscale

		int height = src.rows;
		int width = src.cols;
		int hist[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		imshow("gray image", src);
		showHistogram("histogram", hist, width, height);
		waitKey();
		return hist;
	}
}

float* testFDP(Mat src, int width, int height) {
	int hist[256] = { 0 };
	float fdp[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[src.at<uchar>(i, j)]++;
		}
	}
	for (int i = 0; i < 256; i++)
		fdp[i] = (float)hist[i] / (width * height);

	return fdp;
}

void multipleTresholds() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		imshow("Initial", img);
		int height = img.rows;
		int width = img.cols;

		float* FDP = testFDP(img, width, height);

		int* maxHist = (int*)calloc(256, 4);;
		int WH = 5;
		int widthWindow = 2 * WH + 1;
		float TH = 0.0003, v;
		bool ok;

		for (int k = 0 + WH; k <= 255 - WH; k++) {
			v = 0;
			ok = true;
			for (int i = k - WH; i <= k + WH; i++) {
				v = v + FDP[i];
				if (FDP[k] < FDP[i]) {
					ok = false;
				}
			}
			v = v / (float)widthWindow;

			if (FDP[k] > v + TH && ok) {
				maxHist[k] = k;
				printf("%d %d\n", k, maxHist[k]);
			}
		}
		maxHist[0] = 0;
		maxHist[255] = 255;

		int max, pos;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				max = 999;
				for (int t = 0; t < 256; t++) {
					if (abs(img.data[i * img.step + j] - maxHist[t]) <= max) {
						pos = t;
						max = img.data[i * img.step + j] - maxHist[t];
					}
				}
				img.data[i * img.step + j] = maxHist[pos];
			}
		}
		imshow("Tresholding", img);
		waitKey(0);
	}
}

void floydSteinberg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		imshow("Initial", img);
		int height = img.rows;
		int width = img.cols;
		Mat m1 = Mat(height, width, CV_8UC1);
		Mat m2 = Mat(height, width, CV_8UC1);
		float* FDP = testFDP(img, width, height);

		int* maxHist = (int*)calloc(256, 4);;
		int WH = 5;
		int widthWindow = 2 * WH + 1;
		float TH = 0.0003, v;
		bool ok;

		for (int k = 0 + WH; k <= 255 - WH; k++) {
			v = 0;
			ok = true;
			for (int i = k - WH; i <= k + WH; i++) {
				v = v + FDP[i];
				if (FDP[k] < FDP[i]) {
					ok = false;
				}
			}
			v = v / (float)widthWindow;

			if (FDP[k] > v + TH && ok) {
				maxHist[k] = k;
				printf("%d %d\n", k, maxHist[k]);
			}
		}
		maxHist[0] = 0;
		maxHist[255] = 255;

		int max, pos;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				max = 999;
				for (int t = 0; t < 256; t++) {
					if (abs(img.data[i * img.step + j] - maxHist[t]) <= max) {
						pos = t;
						max = img.data[i * img.step + j] - maxHist[t];
					}
				}
				m1.data[i * m1.step + j] = maxHist[pos];
			}
		}
		imshow("Multiple", m1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				m2.at<uchar>(i, j) = m1.at<uchar>(i, j);
				int eroare = img.at<uchar>(i, j) - m1.at<uchar>(i, j);
				m2.at<uchar>(i + 1, j) += eroare * 7 / 16;
				m2.at<uchar>(i - 1, j + 1) += eroare * 3 / 16;
				m2.at<uchar>(i, j + 1) += eroare * 5 / 16;
				m2.at<uchar>(i + 1, j + 1) += eroare * 16;

			}
		}

		imshow("Floyd-Steinberg", m2);
		waitKey(0);
	}
}

int CalculateAria(int x, int y, int width, int height, Mat* src) {
	int arie = 0;

	int R = (int)(*src).at<Vec3b>(y, x)[2];
	int G = (int)(*src).at<Vec3b>(y, x)[1];
	int B = (int)(*src).at<Vec3b>(y, x)[0];

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			int R1 = (int)(*src).at<Vec3b>(i, j)[2];
			int G1 = (int)(*src).at<Vec3b>(i, j)[1];
			int B1 = (int)(*src).at<Vec3b>(i, j)[0];
			if (R1 == R && G1 == G && B1 == B)
			{
				arie++;
			}
		}
	}

	return arie;
}

void CalculateProperties(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		int width = (*src).cols;
		int height = (*src).rows;

		int R = (int)(*src).at<Vec3b>(y, x)[2];
		int G = (int)(*src).at<Vec3b>(y, x)[1];
		int B = (int)(*src).at<Vec3b>(y, x)[0];

		if (R == 255 && G == 255 && B == 255)
		{
			printf("Not an object");
			return;
		}

		int arie = CalculateAria(x, y, width, height, src);
		printf("Aria = %d\n", arie);

		//Perimetru
		int p = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int R1 = (int)(*src).at<Vec3b>(i, j)[2];
				int G1 = (int)(*src).at<Vec3b>(i, j)[1];
				int B1 = (int)(*src).at<Vec3b>(i, j)[0];

				if (R1 == R && G1 == G && B1 == B) {
					int ok = 0;
					if (isInside(*src, i - 1, j - 1)) {
						int R2 = (int)(*src).at<Vec3b>(i - 1, j - 1)[2];
						int G2 = (int)(*src).at<Vec3b>(i - 1, j - 1)[1];
						int B2 = (int)(*src).at<Vec3b>(i - 1, j - 1)[0];

						if (R1 != R2 || G1 != G2 || B1 != B2) {
							ok = 1;
						}
					}

					if (isInside(*src, i, j - 1)) {
						int R2 = (int)(*src).at<Vec3b>(i, j - 1)[2];
						int G2 = (int)(*src).at<Vec3b>(i, j - 1)[1];
						int B2 = (int)(*src).at<Vec3b>(i, j - 1)[0];

						if (R1 != R2 || G1 != G2 || B1 != B2) {
							ok = 1;
						}
					}

					if (isInside(*src, i + 1, j)) {
						int R2 = (int)(*src).at<Vec3b>(i + 1, j)[2];
						int G2 = (int)(*src).at<Vec3b>(i + 1, j)[1];
						int B2 = (int)(*src).at<Vec3b>(i + 1, j)[0];

						if (R1 != R2 || G1 != G2 || B1 != B2) {
							ok = 1;
						}
					}

					if (isInside(*src, i + 1, j + 1)) {
						int R2 = (int)(*src).at<Vec3b>(i + 1, j + 1)[2];
						int G2 = (int)(*src).at<Vec3b>(i + 1, j + 1)[1];
						int B2 = (int)(*src).at<Vec3b>(i + 1, j + 1)[0];

						if (R1 != R2 || G1 != G2 || B1 != B2) {
							ok = 1;
						}
					}

					if (ok == 1) {
						p++;
					}
				}
			}
		}

		printf("Perimetru = %d\n", p);

		//Centrul De Masa
		int r = 0;
		int c = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int R1 = (int)(*src).at<Vec3b>(i, j)[2];
				int G1 = (int)(*src).at<Vec3b>(i, j)[1];
				int B1 = (int)(*src).at<Vec3b>(i, j)[0];

				if (R1 == R && G1 == G && B1 == B) {
					r += i;
					c += j;
				}
			}
		}

		r = r / arie;
		c = c / arie;

		Mat dst = Mat(height, width, CV_8UC3);
		dst.at<Vec3b>(r, c) = { 0, 0, 0 };
		dst.at<Vec3b>(r + 1, c) = { 0, 0, 0 };
		dst.at<Vec3b>(r, c + 1) = { 0, 0, 0 };
		dst.at<Vec3b>(r - 1, c) = { 0, 0, 0 };
		dst.at<Vec3b>(r, c - 1) = { 0, 0, 0 };

		printf("r = %d c = %d\n", r, c);

		//Axa de alungire

		float rc = 0;
		float r2 = 0;
		float c2 = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int R1 = (int)(*src).at<Vec3b>(i, j)[2];
				int G1 = (int)(*src).at<Vec3b>(i, j)[1];
				int B1 = (int)(*src).at<Vec3b>(i, j)[0];

				if (R1 == R && G1 == G && B1 == B) {
					rc += (i - r) * (j - c);
					r2 += (j - c) * (j - c);
					c2 += (i - r) * (i - r);
				}
			}
		}

		rc *= 2;
		float lungimeDeAxa = atan2(rc, r2 - c2) / 2.0;

		printf("Lungime de Axa = %f\n", lungimeDeAxa);

		float thinnesRatio = 4 * PI * (float)arie / (float)(p * p);

		printf("Thinnes Ratio = %f\n", thinnesRatio);

		int cmax = 0;
		int cmin = x;
		int rmax = 0;
		int rmin = y;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int R1 = (int)(*src).at<Vec3b>(i, j)[2];
				int G1 = (int)(*src).at<Vec3b>(i, j)[1];
				int B1 = (int)(*src).at<Vec3b>(i, j)[0];

				if (R1 == R && G1 == G && B1 == B) {
					if (i > rmax) {
						rmax = i;
					}

					if (i < rmin) {
						rmin = i;
					}

					if (j > cmax) {
						cmax = j;
					}

					if (j < cmin) {
						cmin = j;
					}
				}
			}
		}

		printf("%d %d %d %d\n", rmax, rmin, cmax, cmin);

		float tmp1 = rmax - rmin + 1;
		float tmp2 = cmax - cmin + 1;

		float aspectRatio = tmp2 / tmp1;

		printf("Aspect Ratio = %f\n", aspectRatio);

		Vec3b obj_color = (*src).at<Vec3b>(y, x);

		// Contour
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b current_color = (*src).at<Vec3b>(i, j);

				if (current_color == obj_color) {
					int ok = 0;
					if (isInside(*src, i - 1, j - 1)) {
						Vec3b color = (*src).at<Vec3b>(i - 1, j - 1);
						if (color != obj_color) {
							ok = 1;
						}
					}

					if (isInside(*src, i, j - 1)) {
						Vec3b color = (*src).at<Vec3b>(i, j - 1);
						if (color != obj_color) {
							ok = 1;
						}
					}

					if (isInside(*src, i + 1, j)) {
						Vec3b color = (*src).at<Vec3b>(i + 1, j);
						if (color != obj_color) {
							ok = 1;
						}
					}

					if (isInside(*src, i + 1, j + 1)) {
						Vec3b color = (*src).at<Vec3b>(i + 1, j + 1);
						if (color != obj_color) {
							ok = 1;
						}
					}

					if (ok == 1) {
						dst.at<Vec3b>(i, j) = { 0, 0, 0 };
					}
				}
			}
		}

		Mat projection_horizontal = Mat(height, width, CV_8UC3);
		Mat projection_vertical = Mat(height, width, CV_8UC3);

		int horizontalProjection[1000];
		int verticalProjection[1000];
		int tmpVertical = 0;
		int tmpHorizontal = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if ((*src).at<Vec3b>(i, j) == obj_color) {
					tmpHorizontal++;
				}
			}
			horizontalProjection[i] = tmpHorizontal;
			tmpHorizontal = 0;
		}

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if ((*src).at<Vec3b>(j, i) == obj_color) {
					tmpVertical++;
				}
			}
			verticalProjection[i] = tmpVertical;
			tmpVertical = 0;
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < horizontalProjection[i]; j++) {
				projection_horizontal.at<Vec3b>(i, j) = obj_color;
			}
		}

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < verticalProjection[i]; j++) {
				projection_vertical.at<Vec3b>(j, i) = obj_color;
			}
		}

		imshow("Horizontal Projection", projection_horizontal);
		imshow("Vertical Projection", projection_vertical);
		imshow("Center of Mass, Contour", dst);
	}
}

void onMouse()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", CalculateProperties, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void widthTraversal(int vecini) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Initial", src);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3);
		Mat labels = Mat(height, width, CV_32SC1);
		int label = 0;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				labels.at<int>(i, j) = 0;
			}
		int di[] = { -1,-1,-1, 0,0, 1,1,1 };
		int dj[] = { -1, 0, 1,-1,1,-1,0,1 };
		int fi[] = { -1, 1, -1, 0 };
		int fj[] = { -1, 0, 0, 1 };

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (src.at<uchar>(i, j) == 0 &&
					labels.at<int>(i, j) == 0) {
					label++;
					labels.at<int>(i, j) = label;
					std::queue<Point> coada;
					coada.push(Point(i, j));
					while (!coada.empty()) {
						Point p = coada.front();
						coada.pop();
						int x = p.x;
						int y = p.y;
						if (vecini == 8) {
							for (int k = 0; k < 8; k++) {
								int x1 = x + di[k];
								int y1 = y + dj[k];

								if (isInside(src, x1, y1) && src.at<uchar>(x1, y1) == 0 && labels.at<int>(x1, y1) == 0) {
									labels.at<int>(x1, y1) = label;
									coada.push(Point(x1, y1));
								}
							}
						}
						else {
							for (int k = 0; k < 4; k++) {
								int x1 = x + fi[k];
								int y1 = y + fj[k];

								if (isInside(src, x1, y1) && src.at<uchar>(x1, y1) == 0 && labels.at<int>(x1, y1) == 0) {
									labels.at<int>(x1, y1) = label;
									coada.push(Point(x1, y1));
								}
							}
						}
					}
				}
		std::vector<Vec3b> culori(label + 1);
		for (int i = 1; i <= label; i++) {
			culori[i] = { (unsigned char)d(gen), (unsigned char)d(gen), (unsigned char)d(gen) };
		}
		std::cout << label << std::endl;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<int>(i, j) == 0) {
					dst.at<Vec3b>(i, j) = { 255,255,255 };
				}
				else
					dst.at<Vec3b>(i, j) = culori[labels.at<int>(i, j)];
			}
		}
		imshow("Final", dst);
		waitKey(0);

	}
}

void secondTraversal(int vecini) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Initial", src);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3);
		Mat labels = Mat(height, width, CV_32SC1);
		int label = 0;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				labels.at<int>(i, j) = 0;
			}
		std::vector<std::vector<int>> edges(10000);
		int di[] = { -1,-1,-1, 0,0, 1,1,1 };
		int dj[] = { -1, 0, 1,-1,1,-1,0,1 };
		int fi[] = { -1, 1, -1, 0 };
		int fj[] = { -1, 0, 0, 1 };

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					std::vector<int> L;
					Point n = Point(i, j);
					labels.at<int>(i, j) = label;
					if (vecini == 8) {
						for (int k = 0; k < 8; k++) {
							int x1 = n.x + di[k];
							int y1 = n.y + dj[k];

							if (isInside(src, x1, y1) && src.at<uchar>(x1, y1) == 0 && labels.at<int>(x1, y1) > 0) {
								L.push_back(labels.at<int>(x1, y1));
							}
						}
					}
					else {
						for (int k = 0; k < 4; k++) {
							int x1 = n.x + fi[k];
							int y1 = n.y + fj[k];

							if (isInside(src, x1, y1) && src.at<uchar>(x1, y1) == 0 && labels.at<int>(x1, y1) > 0) {
								L.push_back(labels.at<int>(x1, y1));
							}
						}
					}
					if (L.size() == 0) {
						label++;
						labels.at<int>(i, j) = label;
					}
					else {
						int x = 999;
						for (int i = 0; i < L.size(); i++) {
							if (L[i] < x) {
								x = L[i];
							}
						}
						labels.at<int>(i, j) = x;
						for (auto y : L) {
							if (y != x) {
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}
					}
				}
			}
		std::vector<int> newlabels(label + 1, 0);
		int newlabel = 0;

		for (int i = 1; i <= label; i++) {
			if (newlabels[i] == 0) {
				newlabel++;
				std::queue<int> q;
				newlabels[i] = newlabel;
				q.push(i);
				while (!q.empty()) {
					int x = q.front();
					q.pop();
					for (auto y : edges[x]) {
						if (newlabels[y] == 0) {
							newlabels[y] = newlabel;
							q.push(y);
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<int>(i, j) != 0) {
					labels.at<int>(i, j) = newlabels[labels.at<int>(i, j)];
				}
			}
		}

		std::vector<Vec3b> culori(label + 1);
		for (int i = 1; i <= label; i++) {
			culori[i] = { (unsigned char)d(gen), (unsigned char)d(gen), (unsigned char)d(gen) };
		}
		std::cout << label << std::endl;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<int>(i, j) == 0) {
					dst.at<Vec3b>(i, j) = { 255,255,255 };
				}
				else
					dst.at<Vec3b>(i, j) = culori[labels.at<int>(i, j)];
			}
		}
		imshow("Final", dst);
		waitKey(0);

	}
}

void contourCod() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		int di[] = { 0,-1,-1,-1,0,1,1,1 };
		int dj[] = { 1,1,0,-1,-1,-1,0,1 };

		uchar pi = 0, pj = 0;

		Point pstart0, pstart1;
		Point panterior, pcurent;

		for (int i = 0; i < height; i++) {
			int ok = 0;
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0) {
					pi = i;
					pj = j;
					ok = 1;
					break;
				}
			}
			if (ok == 1) {
				break;
			}
		}

		pstart0.x = pi;
		pstart0.y = pj;

		std::vector<int> dir;
		std::vector<int> der;

		int d = 7;
		int derivat;

		if (d % 2 == 1) {
			d = (d + 6) % 8;
		}
		else {
			d = (d + 7) % 8;
		}

		while (isInside(src, pi + di[d], pj + dj[d]) && src.at<uchar>(pi + di[d], pj + dj[d]) == 255) {
			d++;
			if (d == 8) {
				d = 0;
			}
		}

		pi = pi + di[d];
		pj = pj + dj[d];

		pstart1.x = pi;
		pstart1.y = pj;

		dir.push_back(d);

		pcurent = pstart1;
		panterior = pstart1;

		int n = 0;

		while (1) {
			if (d % 2 == 1) {
				d = (d + 6) % 8;
			}
			else {
				d = (d + 7) % 8;
			}

			while (isInside(src, pi + di[d], pj + dj[d]) && src.at<uchar>(pi + di[d], pj + dj[d]) != 0) {
				d++;
				if (d == 8) {
					d = 0;
				}
			}

			pi = pi + di[d];
			pj = pj + dj[d];

			panterior = pcurent;

			pcurent.x = pi;
			pcurent.y = pj;

			dir.push_back(d);
			n++;

			derivat = (dir[n] - dir[n - 1] + 8) % 8;
			der.push_back(derivat);

			if (pcurent == pstart1 && panterior == pstart0) {
				break;
			}

		}

		dir.pop_back();
		dir.pop_back();

		der.pop_back();
		der.pop_back();

		for (int i = 0; i < dir.size(); i++) {
			printf("%d ", dir[i]);
		}
		printf("\n\n");

		for (int i = 0; i < der.size(); i++) {
			printf("%d ", der[i]);
		}

		printf("\n\n");

		Mat direct = Mat(height, width, CV_8UC1);
		Mat derivata = Mat(height, width, CV_8UC1);

		int x = pstart1.x;
		int y = pstart1.y;

		direct.at<uchar>(x, y) = 0;
		derivata.at<uchar>(x, y) = 0;

		for (int i = 0; i < dir.size(); i++) {
			direct.at<uchar>(x + di[dir[i]], y + dj[dir[i]]) = 0;
			x = x + di[dir[i]];
			y = y + dj[dir[i]];
		}

		x = pstart1.x;
		y = pstart1.y;

		for (int i = 0; i < der.size(); i++) {
			derivata.at<uchar>(x + di[der[i]], y + dj[der[i]]) = 0;
			x = x + di[der[i]];
			y = y + dj[der[i]];
		}

		imshow("direct", direct);
		imshow("derivata", derivata);
		waitKey(0);


	}


}

void reconstructContour() {
	char ch, file_name[25];
	FILE* fp;

	fp = fopen("reconstruct.txt", "r");

	if (fp == NULL) {
		perror("File not found \n");
		exit(EXIT_FAILURE);
	}

	int x, y;
	int size;

	fscanf(fp, "%d  %d ", &x, &y);
	fscanf(fp, "%d", &size);

	int* contour = (int*)calloc(size + 1, sizeof(int));
	int n = 0;

	while (n <= size) {
		int aux = 0;
		fscanf(fp, "%d", &aux);
		contour[n] = aux;
		n++;
	}

	fclose(fp);

	Mat reconstruct = Mat(750, 750, CV_8UC1);

	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	for (int i = 0; i < size; i++) {
		reconstruct.at<uchar>(x + di[contour[i]], y + dj[contour[i]]) = 0;
		x = x + di[contour[i]];
		y = y + dj[contour[i]];
	}

	imshow("output image", reconstruct);
	waitKey();
}

/*Dilatare:
1. Daca suprapunem cu centrul sablonului un punct de
fundal, il ignoram.
2. Daca suprapunem un punct de obiect, ii transformam
vecinii (ce acopera sablonul) in puncte obiect.*/

Mat dilate(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	int x, y;

	Mat dst = src.clone();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++) {
					x = i + di[k];
					y = j + dj[k];
					if (isInside(src, x, y)) {
						dst.at<uchar>(x, y) = 0;
					}
				}
			}
		}
	}
	return dst;
}

void dilateN(int n) {

	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = src.clone();
	Mat temp = dst.clone();
	int i = 0;
	for (i = 0; i < n; i++) {
		temp = dilate(dst);
		dst = temp.clone();
	}

	imshow("Original", src);
	imshow("Dilate", dst);
	waitKey();

}

Mat erode(Mat src)
{
	Mat dst = src.clone();

	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;
	int x = 0;
	int y = 0;
	int k = 0;

	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				for (k = 0; k < 8; k++)
				{
					x = i + di[k];
					y = j + dj[k];
					if (isInside(src, x, y))
					{
						if (src.at<uchar>(x, y) == 255)
						{
							dst.at<uchar>(i, j) = 255;
						}
					}
				}
			}
		}
	}

	return dst;
}

void erodeN(int n) {

	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = src.clone();
	Mat temp = dst.clone();
	int i = 0;
	for (i = 0; i < n; i++) {
		temp = erode(dst);
		dst = temp.clone();
	}

	imshow("Original", src);
	imshow("Erode", dst);
	waitKey();

}

// open = erode + dilate

void openN(int n) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = src.clone();
	Mat temp = dst.clone();
	for (int i = 0; i < n; i++) {
		temp = erode(dst);
		dst = dilate(temp);
	}
	imshow("Original", src);
	imshow("Open", dst);
	waitKey();
}
// close = dilate + erode

Mat close(Mat src) {

	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;
	int x = 0;
	int y = 0;
	int k = 0;

	Mat dst(height, width, CV_8UC1, Scalar(255));

	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				for (k = 0; k < 8; k++)
				{
					x = i + di[k];
					y = j + dj[k];
					if (isInside(src, x, y))
					{
						dst.at<uchar>(x, y) = 0;
					}
				}
			}
		}
	}

	Mat dstcpy = dst.clone();

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (dstcpy.at<uchar>(i, j) == 0)
			{
				for (k = 0; k < 8; k++)
				{
					x = i + di[k];
					y = j + dj[k];
					if (isInside(dstcpy, x, y))
					{
						if (src.at<uchar>(x, y) == 255 && dstcpy.at<uchar>(x,y) == 255)
						{
							dst.at<uchar>(i, j) = 255;
						}
					}
				}
			}
		}
	}

	return dst;
}

void closeN(int n) {

	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = src.clone();
	Mat temp = dst.clone();
	int i = 0;
	for (i = 0; i < n; i++) {
		temp = close(dst);
		dst = temp.clone();
	}

	imshow("Original", src);
	imshow("Close", dst);
	waitKey();

}

void contour() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = src.clone();

	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;
	int x = 0;
	int y = 0;

	imshow("Original", src);

	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				for (int k = 0; k < 8; k++)
				{
					x = i + di[k];
					y = j + dj[k];
					if (isInside(src, x, y))
					{
						if (src.at<uchar>(x, y) == 255)
						{
							dst.at<uchar>(i, j) = 255;
						}
					}
				}
			}
		}
	}


	Mat dstcpy = dst.clone();
	dst = src.clone();

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == dstcpy.at<uchar>(i, j))
			{
				dst.at<uchar>(i, j) = 255;
			}
			else dst.at<uchar>(i, j) = src.at<uchar>(i, j);
		}
	}
	imshow("Contur", dst);
	waitKey();
}

/*Region filling:
1. Stocam complementul intr-o noua imagine
2. Luam un punct pe mijloc intr-o imagine noua
Do
3. Dilata punctul
4. Intersecteaza dilatarea cu complementul
While (ultima dilatare != penultima dilatare)
5. Reunim imaginea sursa cu imaginea rezultat
(adaugam conturul la rezultat)*/

Mat intersect(Mat src, Mat dst) {
	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0 && dst.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = 0;
			}
			else dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

Mat reunion(Mat src, Mat dst) {
	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0 || dst.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = 0;
			}
			else dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

bool compare(Mat src, Mat dst) {
	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) != dst.at<uchar>(i, j))
			{
				return false;
			}
		}
	}
	return true;
}

void regionFilling() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat dst = src.clone();
	Mat complement = src.clone();
	int height = src.rows;
	int width = src.cols;
	imshow("Original", src);

	// complement:
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				complement.at<uchar>(i, j) = 255;
			}
			else complement.at<uchar>(i, j) = 0;
		}
	}
	imshow("Complement", complement);

	Mat img = Mat(height, width, CV_8UC1, Scalar(255));
	int x = height / 2;
	int y = width / 2;
	img.at<uchar>(x, y) = 0;
	Mat lastDilate, currDilate = img.clone();
	do {
		lastDilate = currDilate;
		currDilate = dilate(currDilate);
		currDilate = intersect(complement, currDilate);
	} while (!compare(currDilate, lastDilate));
	imshow("Dilate & intersect", currDilate);

	img = reunion(img, currDilate);
	imshow("Region filling", img);
	waitKey();
}

void cumulativeHistogram(){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	int m = width * height;
	float intensity = 0;
	float dev = 0;
	int i = 0;
	int j = 0;

	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[src.at<uchar>(i, j)]++;
		}
	}
	showHistogram("histogram", hist, width, height);

	int cum_hist[256] = { 0 };
	cum_hist[0] = hist[0];
	for (int i = 1; i < 256; i++)
	{
		cum_hist[i] = cum_hist[i - 1] + hist[i];
	}
	showHistogram("cumulative histogram", cum_hist, width, height);

	//intensity 
	for (i = 0; i < 256; i++)
	{
		intensity += i * hist[i];
	}
	intensity /= m;

	//deviation
	for (i = 0; i < 256; i++)
	{
		dev += pow(i - intensity, 2) * hist[i]/m;
	}
	dev = sqrt(dev);

	printf("Intensity: %f /n", intensity);
	printf("Deviation: %f /n", dev);
	waitKey();

}

void autoBinary(){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	int m = width * height;
	int i = 0;
	int j = 0;

	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[src.at<uchar>(i, j)]++;
		}
	}
	showHistogram("histogram", hist, width, height);

	int iMax, iMin;

	for (i = 0; i < 256; i++)
	{
		if (hist[i] != 0)
		{
			iMin = i;
			break;
		}
	}
	for (i = 255; i >= 0; i--)
	{
		if (hist[i] != 0)
		{
			iMax = i;
			break;
		}
	}

	float T1 = (iMax + iMin) / 2;
	float T2 = 0;
	float err = 0.1;
	float u1,u2;
	int n1,n2;

	while (abs(T1 - T2) > err) {
		u1 = 0;
		u2 = 0;
		n1 = 0;
		n2 = 0;
		i = iMin;
		while ( i <= iMax) {
			if (i <= T1) {
				u1 += i * hist[i];
				n1 += hist[i];
			}
			else {
				u2 += i * hist[i];
				n2 += hist[i];
			}
			i++;
		}
		u1 = u1 / n1;
		u2 = u2 / n2;
		T2 = T1;
		T1 = (u1 + u2) / 2;
	}
	

	printf("Threshold: %f \n", T1);

	Mat dst = src.clone();
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) > T1)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else dst.at<uchar>(i, j) = 0;
		}
	}
	imshow("Binary", dst);
	waitKey();
}

void contrastModification(int gMin, int gMax){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;

	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[src.at<uchar>(i, j)]++;
		}
	}
	showHistogram("histogram", hist, width, height);

	int iMax=0, iMin=0;

	for (i = 0; i < 256; i++)
	{
		if (hist[i] != 0)
		{
			iMin = i;
			break;
		}
	}
	for (i = 255; i >= 0; i--)
	{
		if (hist[i] != 0)
		{
			iMax = i;
			break;
		}
	}

	int modif[256] = { 0 };
	for (i = 0; i < 256; i++)
	{
		modif[i] = gMin + (hist[i] - iMin) * (gMax - gMin) / (iMax - iMin);
	}

	showHistogram("modified histogram", modif, width, height);

	Mat dst = src.clone();
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst.at<uchar>(i, j) = modif[src.at<uchar>(i, j)];
		}
	}
	imshow("Modified", dst);
	waitKey();
}

void gammaCorrection(float gamma){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;

	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[src.at<uchar>(i, j)]++;
		}
	}
	showHistogram("histogram", hist, width, height);

	Mat dst = src.clone();
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst.at<uchar>(i, j) = 255 * pow(src.at<uchar>(i, j) / 255.0, gamma);
			if (dst.at<uchar>(i, j) > 255) dst.at<uchar>(i, j) = 255;
		}
	}

	int modif[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			modif[dst.at<uchar>(i, j)]++;
		}
	}

	showHistogram("modified histogram", modif, width, height);

	imshow("Modified", dst);
	waitKey();

}

void histogramEqualization(){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	int i = 0;
	int j = 0;

	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[src.at<uchar>(i, j)]++;
		}
	}
	showHistogram("histogram", hist, width, height);

	float fdp[256] = { 0 };
	for (int i = 0; i < 256; i++)
		fdp[i] = (float)hist[i] / (width * height);
	float pc[256] = { 0 };
	pc[0] = fdp[0];
	for (int i = 1; i < 256; i++)
		pc[i] = pc[i - 1] + fdp[i];
	
	int modif[256] = { 0 };
	for (int i = 0; i < 256; i++)
		modif[i] = (int)(pc[i] * 255);
	

	Mat dst = src.clone();
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst.at<uchar>(i, j) = modif[src.at<uchar>(i, j)];
		}
	}

	int hist2[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist2[dst.at<uchar>(i, j)]++;
		}
	}
	showHistogram("modified histogram", hist2, width, height);
	imshow("Modified", dst);
	waitKey();
}

void lowPassFilters(int n) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();
	Mat nucleu(3, 3, CV_32FC1);
	if (n == 1) {
		for(int i=0; i<3; i++)
			for (int j = 0; j < 3; j++)
			{
				nucleu.at<float>(i, j) = 1.0;
			}
	}
	else{
		nucleu.at<float>(0,0) = 1.0;
		nucleu.at<float>(0,1) = 2.0;
		nucleu.at<float>(0,2) = 1.0;
		nucleu.at<float>(1,0) = 2.0;
		nucleu.at<float>(1,1) = 4.0;
		nucleu.at<float>(1,2) = 2.0;
		nucleu.at<float>(2,0) = 1.0;
		nucleu.at<float>(2,1) = 2.0;
		nucleu.at<float>(2,2) = 1.0;
	}

	int di[] = { 0,0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 0,1,1,0,-1,-1,-1,0,1 };


	for (int i = 1; i < height-1; i++)
	{
		for (int j = 1; j < width-1; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < 9; k++)
			{
				sum += nucleu.at<float>(1+di[k], 1+dj[k]) * src.at<uchar>(i + di[k], j + dj[k]);
			}
			if (n == 1)
				if(sum/9.0 > 255) dst.at<uchar>(i, j) = 255;
				else dst.at<uchar>(i, j) = sum / 9.0;
			else
				if (sum / 16.0 > 255) dst.at<uchar>(i, j) = 255;
				else dst.at<uchar>(i, j) = sum / 16.0;
		}
	}

	imshow("Modified", dst);
	waitKey();	
}

void highPassFilters(int n) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();
	Mat nucleu(3, 3, CV_32FC1);
	if (n == 1) {
		for (int i = 0; i<3; i++)
			for (int j = 0; j < 3; j++)
			{
				nucleu.at<float>(i, j) = -1.0;
			}
		nucleu.at<float>(1, 1) = 8.0;
	}
	else {
		nucleu.at<float>(0, 0) = -1.0;
		nucleu.at<float>(0, 1) = -1.0;
		nucleu.at<float>(0, 2) = -1.0;
		nucleu.at<float>(1, 0) = -1.0;
		nucleu.at<float>(1, 1) = 9.0;
		nucleu.at<float>(1, 2) = -1.0;
		nucleu.at<float>(2, 0) = -1.0;
		nucleu.at<float>(2, 1) = -1.0;
		nucleu.at<float>(2, 2) = -1.0;
	}

	int di[] = { 0,0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 0,1,1,0,-1,-1,-1,0,1 };

	for (int i = 1; i < height-1; i++)
	{
		for (int j = 1; j < width-1; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < 9; k++)
			{
				sum += nucleu.at<float>(1+di[k], 1+dj[k]) * src.at<uchar>(i + di[k], j + dj[k]);
			}
			if(sum > 255) dst.at<uchar>(i, j) = 255;
			else if (sum < 0) dst.at<uchar>(i, j) = 0;
				else dst.at<uchar>(i, j) = ceil(sum);
		}
	}

	imshow("Modified", dst);
	waitKey();	

}

void centering_transform(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat sablon(Mat src){	
		Mat srcf(src.rows, src.cols, CV_32FC1); //stocam imaginea sursa ca float
		int H = src.rows;
		int W = src.cols;
		src.convertTo(srcf, CV_32FC1);
		
		centering_transform(srcf); //centram imaginea, pentru a o putea procesa in domeniul frecvential
		
		Mat fourier; //cream o matrice de numere complexe in care sa stocam transformata fourier directa
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT); //aplicam transformata fourier directa pe imaginea sursa si rezultatul va fi stocat in matricea "fourier" de numere complexe
		
		//dorim sa desfacem numerele complexe din matricea "fourier" in doua matrici, una cu partile reale si una cu partile imaginare ale numerelor 
		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) }; 
		split(fourier, channels);//separam partea reala de partea imaginara
		//in matricea channels[0] avem matricea care contine partile reale ale numerelor din "fourier"
		//in matricea channels[1] avem partea imaginara a numerelor
		
				//calculul magnitudinii
				Mat mag;
				magnitude(channels[0], channels[1], mag);
				//parcurgem matricea "mag" si aplicam log(mag.at<float>(i, j) + 1) pe fiecare element
				//aplicam functia normalize pe matricea mag si o afisam
				//calculul fazei
				Mat phi;
				phase(channels[0], channels[1], phi);
		
			//aici aplicam filtre (R=20 sau A=20)
			//parcurgem imaginea (0->H, 0->W) si aplicam modificarile pe channels[0] si channels[1]
			//pentru gauss putem folosi functia exp(exponent)
			
				//exemplu pentru filtru gaussian low pass
				int A=20;
				for (int i=0; i<H; i++){
					for (int j=0; j<W; j++){
						channels[0].at<float>(i, j) *= exp(-((H/2-i)*(H/2-i) + (W/2-j)*(W/2-j))/(A*A));
						channels[1].at<float>(i, j) *= exp(-((H/2-i)*(H/2-i) + (W/2-j)*(W/2-j))/(A*A));
					}
				}
	
		Mat dst, dstf;
		merge(channels, 2, fourier); //reunim cele doua canale (real si imaginar), pentru a pregati matricea de revenirea in domeniul spatial
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE); //aplicam transformata fourier inversa pentru a reveni in domeniul spatial
		//in matricea dstf avem rezultatul transformatei fourier inverse
		
		centering_transform(dstf); //recentram imaginea, pentru a o vizualiza usor in domeniul spatial
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1); //normalizam matricea, ca sa nu avem valori in afara intervalului 0-255; punem rezultatul in dst
		
		return dst;
}

void magnitudeCalculator(){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat srcf(src.rows, src.cols, CV_32FC1); //stocam imaginea sursa ca float
	int H = src.rows;
	int W = src.cols;
	src.convertTo(srcf, CV_32FC1);
	
	centering_transform(srcf); 
	
	Mat fourier; 
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT); 
	
	Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) }; 
	split(fourier, channels);

	Mat mag, magOut;
	magnitude(channels[0], channels[1], mag);
	for(int i=0; i<H; i++)
		for(int j=0; j<W; j++)
			mag.at<float>(i, j) = log(mag.at<float>(i, j) + 1); //aplicam log(mag.at<float>(i, j) + 1) pe fiecare element din matricea "mag"
	normalize(mag, magOut, 0, 255, NORM_MINMAX, CV_8UC1); //normalizam matricea, ca sa nu avem valori in afara intervalului 0-255; punem rezultatul in dst

	Mat phi;
	phase(channels[0], channels[1], phi);

	imshow("Original", src);
	imshow("Magnitude", magOut);
	imshow("Phase", phi);
	waitKey();
}

void lowPassFrequential(int n){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat srcf(src.rows, src.cols, CV_32FC1); //stocam imaginea sursa ca float
	int H = src.rows;
	int W = src.cols;
	src.convertTo(srcf, CV_32FC1);

	centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
	split(fourier, channels);
	if(n==1){
		int R = 20;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				if ((i - H / 2) * (i - H / 2) + (j - W / 2) * (j - W / 2) > R*R) {
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}
	}
	else{
		int A = 20;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				channels[0].at<float>(i, j) *= exp(-((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) / (A * A));
				channels[1].at<float>(i, j) *= exp(-((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) / (A * A));
			}
		}
	}
	

	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	centering_transform(dstf); //recentram imaginea, pentru a o vizualiza usor in domeniul spatial
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1); //normalizam matricea, ca sa nu avem valori in afara intervalului 0-255; punem rezultatul in dst

	imshow("Original", src);
	imshow("Modified", dst);
	waitKey();
}

void highPassFrequential(int n) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	Mat srcf(src.rows, src.cols, CV_32FC1); //stocam imaginea sursa ca float
	int H = src.rows;
	int W = src.cols;
	src.convertTo(srcf, CV_32FC1);

	centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
	split(fourier, channels);
	if (n == 1) {
		int R = 20;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				if ((H/2-i) * (H / 2 - i) + (W/2-j) * (W / 2 - j) <= R * R) {
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}
	}
	else {
		int A = 20;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				channels[0].at<float>(i, j) *= (1-exp(-((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) / (A * A)));
				channels[1].at<float>(i, j) *= (1-exp(-((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) / (A * A)));
			}
		}
	}


	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	centering_transform(dstf); //recentram imaginea, pentru a o vizualiza usor in domeniul spatial
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1); //normalizam matricea, ca sa nu avem valori in afara intervalului 0-255; punem rezultatul in dst

	imshow("Original", src);
	imshow("Modified", dst);
	waitKey();
}

//salt and pepper filters: median filter, minimal filter, maximal filter

void saltAndPepper() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	Mat median = Mat(height, width, CV_8UC1);
	Mat minimal = Mat(height, width, CV_8UC1);
	Mat maximal = Mat(height, width, CV_8UC1);

	int di[] = { 0,0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 0,1,1,0,-1,-1,-1,0,1 };

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			std::vector<int> v;
			for (int k = 0; k < 9; k++) {
				v.push_back(src.at<uchar>(i + di[k], j + dj[k]));
			}
			sort(v.begin(), v.end());
			median.at<uchar>(i, j) = v[4];
			minimal.at<uchar>(i, j) = v[0];
			maximal.at<uchar>(i, j) = v[8];
		}
	}

	imshow("Median", median);
	imshow("Minimal", minimal);
	imshow("Maximal", maximal);
	waitKey();
}

void gaussianNoiseFilter(){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;
	float sigma = 0.8;
	// dimensiunea filtrului e 6*sigma
	int w = ceil(6 * sigma);
	//x0, y0 sunt mijlocul filtrului
	int x0 = w / 2;
	int y0 = w / 2;

	// Mat filter method
	double t1 = (double)getTickCount();
	Mat gaussianMat = src.clone();

	// filtru: exp(-(x-x0)^2 + (y-y0)^2 / 2*sigma^2) / 2*pi*sigma^2
	Mat filtru = Mat(w, w, CV_32FC1);
	float sum = 0;
	
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			filtru.at<float>(i, j) = exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * sigma * sigma)) / (2 * 3.1415 * sigma * sigma);
			sum += filtru.at<float>(i, j);
		}
	}

	for (int i = x0; i < height - x0; i++) {
		for (int j = y0; j < width - y0; j++) {
			float rez = 0;
			for (int k = 0; k < w; k++) {
				for (int l = 0; l < w; l++) {
					rez += filtru.at<float>(k, l) * src.at<uchar>(i + k - x0, j + l - y0);
				}
			}
			gaussianMat.at<uchar>(i, j) = rez/sum;
		}
	}
	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	printf("Time Gaussian Mat= %.3f [s]\n", t1);

	// 2 filter vectors method
	double t2 = (double)getTickCount();
	Mat gaussianVec = src.clone();
	Mat gaussianAux = src.clone();
	std::vector<float> gx(w);
	std::vector<float> gy(w);
	float sumx = 0;
	float sumy = 0;

	for (int i = 0; i < w; i++) {
		gx[i] = exp(-(i - x0) * (i - x0) / (2 * sigma * sigma)) / (sqrt(2 * 3.1415) * sigma);
		gy[i] = exp(-(i - y0) * (i - y0) / (2 * sigma * sigma)) / (sqrt(2 * 3.1415) * sigma);
		sumx += gx[i];
		sumy += gy[i];
	}

	for (int i = x0; i < height - x0; i++) {
		for (int j = y0; j < width - y0; j++) {
			float rez = 0;
			for (int k = 0; k < w; k++) {
				rez += gx[k] * src.at<uchar>(i + k - x0, j);
			}
			gaussianAux.at<uchar>(i, j) = rez / sumx;
		}
	}

	for (int i = x0; i < height - x0; i++) {
		for (int j = y0; j < width - y0; j++) {
			float rez = 0;
			for (int k = 0; k < w; k++) {
				rez += gy[k] * gaussianAux.at<uchar>(i, j + k - y0);
			}
			gaussianVec.at<uchar>(i, j) = rez / sumy;
		}
	}
	t2 = ((double)getTickCount() - t2) / getTickFrequency();
	printf("Time Gaussian Vec= %.3f [s]\n", t2);

	imshow("Gaussian Mat", gaussianMat);
	imshow("Gaussian Vec", gaussianVec);
	waitKey();
}

void canny(){
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	imshow("Original", src);
	int height = src.rows;
	int width = src.cols;

	// step 1: Filtrarea imaginii cu filtru Gaussian
	float sigma = 0.8;
	// dimensiunea filtrului e 6*sigma
	int w = ceil(6 * sigma);
	//x0, y0 sunt mijlocul filtrului
	int x0 = w / 2;
	int y0 = w / 2;

	Mat gaussianMat = src.clone();

	// filtru: exp(-(x-x0)^2 + (y-y0)^2 / 2*sigma^2) / 2*pi*sigma^2
	Mat filtru = Mat(w, w, CV_32FC1);
	float sum = 0;
	
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			filtru.at<float>(i, j) = exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * sigma * sigma)) / (2 * 3.1415 * sigma * sigma);
			sum += filtru.at<float>(i, j);
		}
	}

	for (int i = x0; i < height - x0; i++) {
		for (int j = y0; j < width - y0; j++) {
			float rez = 0;
			for (int k = 0; k < w; k++) {
				for (int l = 0; l < w; l++) {
					rez += filtru.at<float>(k, l) * src.at<uchar>(i + k - x0, j + l - y0);
				}
			}
			gaussianMat.at<uchar>(i, j) = rez/sum;
		}
	}
	imshow("Gaussian Mat", gaussianMat);

	// step 2: Filtram imaginea cu filtru Sobel
	Mat sobel(3, 3, CV_32FC1);
	sobel.at<float>(0, 0) = 1;
	sobel.at<float>(0, 1) = 2;
	sobel.at<float>(0, 2) = 1;
	sobel.at<float>(1, 0) = 0;
	sobel.at<float>(1, 1) = 0;
	sobel.at<float>(1, 2) = 0;
	sobel.at<float>(2, 0) = -1;
	sobel.at<float>(2, 1) = -2;
	sobel.at<float>(2, 2) = -1;
	
	Mat gx = Mat(height, width, CV_32SC1, Scalar(0));
	Mat gy = Mat(height, width, CV_32SC1, Scalar(0));

	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			int rezx = 0;
			int rezy = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++){
					rezx += sobel.at<float>(2-l, 2-k) * gaussianMat.at<uchar>(i + k - 1, j + l - 1);
					rezy += sobel.at<float>(k,l) * gaussianMat.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			gx.at<int>(i, j) = rezx;
			gy.at<int>(i, j) = rezy;
		}
	}

	Mat modul = Mat(height, width, CV_8UC1, Scalar(0));
	normalize(gaussianMat, modul, 0, 255, NORM_MINMAX, CV_8UC1);
	// direction - mat de float
	Mat directie = Mat(height, width, CV_32F, Scalar(0.0));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			modul.at<uchar>(i, j) = sqrt(pow(gx.at<int>(i, j), 2) + pow(gy.at<int>(i, j), 2)) / (4.0 * sqrt(2));
			directie.at<float>(i, j) = (atan2(gy.at<int>(i, j), gx.at<int>(i, j)) + CV_PI);
		}
	}

	imshow("Modul", modul);

	// step 3: Non-maximum suppression
	Mat nonMax = Mat(height, width, CV_8UC1, Scalar(0));
	normalize(modul, nonMax, 0, 255, NORM_MINMAX, CV_8UC1); 
	
	float ls11 = CV_PI / 8;
	float ld11 = 3 * CV_PI / 8;

	float ls12 = 9 * CV_PI / 8;
	float ld12 = 11 * CV_PI / 8;


	float ls01 = 3 * CV_PI / 8;
	float ld01 = 5 * CV_PI / 8;

	float ls02 = 11 * CV_PI / 8;
	float ld02 = 13 * CV_PI / 8;


	float ls31 = 5 * CV_PI / 8;
	float ld31 = 7 * CV_PI / 8;

	float ls32 = 13 * CV_PI / 8;
	float ld32 = 15 * CV_PI / 8;


	float ls21 = 7 * CV_PI / 8;
	float ld21 = 9 * CV_PI / 8;

	float ls22 = 15 * CV_PI / 8;
	float ld22 = 1 * CV_PI / 8;



	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {

			if (((directie.at<float>(i, j) > ls11) && (directie.at<float>(i, j) <= ld11)) ||
				((directie.at<float>(i, j) > ls12) && (directie.at<float>(i, j) <= ld12))) {

				if (nonMax.at<uchar>(i - 1, j + 1) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i + 1, j - 1) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

			if (((directie.at<float>(i, j) > ls01) && (directie.at<float>(i, j) <= ld01)) ||
				((directie.at<float>(i, j) > ls02) && (directie.at<float>(i, j) <= ld02))) {

				if (nonMax.at<uchar>(i - 1, j) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i + 1, j) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

			if (((directie.at<float>(i, j) > ls31) && (directie.at<float>(i, j) <= ld31)) ||
				((directie.at<float>(i, j) > ls32) && (directie.at<float>(i, j) <= ld32))) {

				if (nonMax.at<uchar>(i - 1, j - 1) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i + 1, j + 1) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

			if (((directie.at<float>(i, j) > ls21) && (directie.at<float>(i, j) <= ld21)) ||
				((directie.at<float>(i, j) > ls22) && (directie.at<float>(i, j) <= ld22))) {

				if (nonMax.at<uchar>(i, j - 1) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i, j + 1) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

		}
	}

	imshow("Non-max", nonMax);
	// Step 4: adaptive binarization
	Mat binarizare = nonMax.clone();
	
	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[binarizare.at<uchar>(i, j)]++;
		}
	}
	showHistogram("modified histogram", hist, width, height);

	int nrPuncteZero = hist[0];
	int nrTotalPuncte = (width-2) * (height-2);
	float p = 0.15;
	int nrPuncteMuchie = p * (nrTotalPuncte - nrPuncteZero);
	int nrNonMuchie = (1-p) * (nrTotalPuncte - nrPuncteZero);

	int treshold = 255;
	sum = 0;
	int x;
	for (x = 1; x < 256; x++) {
		sum += hist[x];
		if (sum > nrNonMuchie) {
			break;
		}
	}
	treshold = x;
	int pragAdaptiv = treshold;
	std::cout << pragAdaptiv << std::endl;
	std::cout << nrTotalPuncte << std::endl;
	std::cout << nrPuncteZero << std::endl;
	std::cout << nrNonMuchie << std::endl;
	std::cout << sum << std::endl;

	// Step 5: Extinderea muchiilor prin histereza
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (binarizare.at<uchar>(i, j) > pragAdaptiv) {
				binarizare.at<uchar>(i, j) = 255;
			}
			else {
				if (binarizare.at<uchar>(i, j) > 0.6*pragAdaptiv) {
					binarizare.at<uchar>(i, j) = 128;
				}
				else {
					binarizare.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	imshow("Binarizare", binarizare);
	//In continuare, vrem sa evidențiem muchiile tari. Daca într-o muchie exista si puncte tari si puncte slabe, le vom face pe cele slabe puncte tari, pentru a nu avea intermediari. Daca o muchie are doar puncte slabe si nu are nici un punct tare, va fi ștearsăcomplet.Putem sa folosim o coada in care sa stocam punctele tari pe care le găsim. Pentru fiecare punct tare căutăm daca exista puncte slabe printre vecinii lui, iar daca exista, le facem tari si le adăugăm in coada.Dupăce ne asiguram ca toate punctele tari au fost extinse in muchiile din care fac parte, parcurgem încăo data imaginea si suprimam (înlocuim cu 0) toate punctele slabe (de intensitate 128), pentru ca ele fac parte din muchii complet slabe si nu mai avem nevoie de ele.
	std::queue<Point> coada;
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (binarizare.at<uchar>(i, j) == 255) {
				coada.push(Point(i, j));
				while (!coada.empty()) {
					Point punct = coada.front();
					coada.pop();
					for (int k = -1; k <= 1; k++) {
						for (int l = -1; l <= 1; l++) {
							if (binarizare.at<uchar>(punct.x + k, punct.y + l) == 128) {
								binarizare.at<uchar>(punct.x + k, punct.y + l) = 255;
								coada.push(Point(punct.x + k, punct.y + l));
							}
						}
					}
				}
			}
		}
	}
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			if (binarizare.at<uchar>(i, j) == 128) {
				binarizare.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("Canny", binarizare);
	waitKey();
}

void calculateLeafProperties(const Mat& binaryImage, double& minArea, double& maxArea, double& minCircularity, double& maxCircularity, std::vector<std::vector<Point>> contours) {
	minArea = (std::numeric_limits<double>::max)();
	maxArea = 0;
	minCircularity = (std::numeric_limits<double>::max)();
	maxCircularity = 0;

	for (const auto& contour : contours) {
		double area = contourArea(contour);
		double perimeter = arcLength(contour, true);
		double circularity = 4 * CV_PI * area / (perimeter * perimeter);

		minArea = (std::min)(minArea, area);
		maxArea = (std::max)(maxArea, area);
		minCircularity = (std::min)(minCircularity, circularity);
		maxCircularity = (std::max)(maxCircularity, circularity);
	}
}

cv::Rect findStemCoordinates(const cv::Mat& binaryImage) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double maxArea = 0;
	int maxAreaIdx = -1;

	for (size_t i = 0; i < contours.size(); ++i) {
		double area = cv::contourArea(contours[i]);
		if (area > maxArea) {
			maxArea = area;
			maxAreaIdx = static_cast<int>(i);
		}
	}

	cv::Rect stemRect;
	if (maxAreaIdx != -1) {
		stemRect = cv::boundingRect(contours[maxAreaIdx]);
	}

	return stemRect;
}

bool hasBlackNeighbor(const cv::Mat& image, int x, int y)
{
	int dx[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	for (int i = 0; i < 8; ++i)
	{
		int nx = x + dx[i];
		int ny = y + dy[i];

		if (nx >= 0 && ny >= 0 && nx < image.cols && ny < image.rows)
		{
			if (image.at<uchar>(ny, nx) == 0)
				return true;
		}
	}

	return false;
}

void traverseContour(const cv::Mat& grayscaleImage, cv::Mat& contourImage, cv::Point startPoint, std::vector<cv::Point>& contour)
{
	int x = startPoint.x;
	int y = startPoint.y;
	if (x < 0 || y < 0 || x >= grayscaleImage.cols || y >= grayscaleImage.rows)
		return;

	contour.push_back(startPoint);
	contourImage.at<uchar>(y, x) = 0;

	int dx[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	for (int i = 0; i < 8; ++i)
	{
		int nx = x + dx[i];
		int ny = y + dy[i];

		if (nx >= 0 && ny >= 0 && nx < grayscaleImage.cols && ny < grayscaleImage.rows)
		{
			if (grayscaleImage.at<uchar>(ny, nx) < 255 && contourImage.at<uchar>(ny, nx) != 0)
			{
				traverseContour(grayscaleImage, contourImage, cv::Point(nx, ny), contour);
			}
		}
	}
}

void findContours2(const cv::Mat& grayscaleImage, std::vector<std::vector<cv::Point>>& contours)
{
	contours.clear();
	cv::Mat contourImage = cv::Mat::zeros(grayscaleImage.size(), CV_8UC1);
	for (int y = 0; y < grayscaleImage.rows; ++y)
	{
		for (int x = 0; x < grayscaleImage.cols; ++x)
		{
			if (grayscaleImage.at<uchar>(y, x) < 255)
			{
				contourImage.at<uchar>(y, x) = 0;
				if (hasBlackNeighbor(contourImage, x, y))
				{
					std::vector<cv::Point> contour;
					cv::Point startPoint(x, y);
					traverseContour(grayscaleImage, contourImage, startPoint, contour);
					contours.push_back(contour);
				}
			}
		}
	}
}

void detectLeaves() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat srcColor = imread(fname, IMREAD_COLOR);
	Mat src;
	cv::cvtColor(srcColor, src, cv::COLOR_BGR2GRAY);
	imshow("Original", srcColor);
	int height = src.rows;
	int width = src.cols;

	// step 1: Filtrarea imaginii cu filtru Gaussian
	float sigma = 0.8;
	// dimensiunea filtrului e 6*sigma
	int w = ceil(6 * sigma);
	//x0, y0 sunt mijlocul filtrului
	int x0 = w / 2;
	int y0 = w / 2;

	Mat gaussianMat = src.clone();

	// filtru: exp(-(x-x0)^2 + (y-y0)^2 / 2*sigma^2) / 2*pi*sigma^2
	Mat filtru = Mat(w, w, CV_32FC1);
	float sum = 0;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			filtru.at<float>(i, j) = exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * sigma * sigma)) / (2 * 3.1415 * sigma * sigma);
			sum += filtru.at<float>(i, j);
		}
	}

	for (int i = x0; i < height - x0; i++) {
		for (int j = y0; j < width - y0; j++) {
			float rez = 0;
			for (int k = 0; k < w; k++) {
				for (int l = 0; l < w; l++) {
					rez += filtru.at<float>(k, l) * src.at<uchar>(i + k - x0, j + l - y0);
				}
			}
			gaussianMat.at<uchar>(i, j) = rez / sum;
		}
	}

	// step 2: Filtram imaginea cu filtru Sobel
	Mat sobel(3, 3, CV_32FC1);
	sobel.at<float>(0, 0) = 1;
	sobel.at<float>(0, 1) = 2;
	sobel.at<float>(0, 2) = 1;
	sobel.at<float>(1, 0) = 0;
	sobel.at<float>(1, 1) = 0;
	sobel.at<float>(1, 2) = 0;
	sobel.at<float>(2, 0) = -1;
	sobel.at<float>(2, 1) = -2;
	sobel.at<float>(2, 2) = -1;

	Mat gx = Mat(height, width, CV_32SC1, Scalar(0));
	Mat gy = Mat(height, width, CV_32SC1, Scalar(0));

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			int rezx = 0;
			int rezy = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					rezx += sobel.at<float>(2 - l, 2 - k) * gaussianMat.at<uchar>(i + k - 1, j + l - 1);
					rezy += sobel.at<float>(k, l) * gaussianMat.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			gx.at<int>(i, j) = rezx;
			gy.at<int>(i, j) = rezy;
		}
	}

	Mat modul = Mat(height, width, CV_8UC1, Scalar(0));
	normalize(gaussianMat, modul, 0, 255, NORM_MINMAX, CV_8UC1);
	// direction - mat de float
	Mat directie = Mat(height, width, CV_32F, Scalar(0.0));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			modul.at<uchar>(i, j) = sqrt(pow(gx.at<int>(i, j), 2) + pow(gy.at<int>(i, j), 2)) / (4.0 * sqrt(2));
			directie.at<float>(i, j) = (atan2(gy.at<int>(i, j), gx.at<int>(i, j)) + CV_PI);
		}
	}

	// step 3: Non-maximum suppression
	Mat nonMax = Mat(height, width, CV_8UC1, Scalar(0));
	normalize(modul, nonMax, 0, 255, NORM_MINMAX, CV_8UC1);

	float ls11 = CV_PI / 8;
	float ld11 = 3 * CV_PI / 8;

	float ls12 = 9 * CV_PI / 8;
	float ld12 = 11 * CV_PI / 8;


	float ls01 = 3 * CV_PI / 8;
	float ld01 = 5 * CV_PI / 8;

	float ls02 = 11 * CV_PI / 8;
	float ld02 = 13 * CV_PI / 8;


	float ls31 = 5 * CV_PI / 8;
	float ld31 = 7 * CV_PI / 8;

	float ls32 = 13 * CV_PI / 8;
	float ld32 = 15 * CV_PI / 8;


	float ls21 = 7 * CV_PI / 8;
	float ld21 = 9 * CV_PI / 8;

	float ls22 = 15 * CV_PI / 8;
	float ld22 = 1 * CV_PI / 8;



	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {

			if (((directie.at<float>(i, j) > ls11) && (directie.at<float>(i, j) <= ld11)) ||
				((directie.at<float>(i, j) > ls12) && (directie.at<float>(i, j) <= ld12))) {

				if (nonMax.at<uchar>(i - 1, j + 1) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i + 1, j - 1) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

			if (((directie.at<float>(i, j) > ls01) && (directie.at<float>(i, j) <= ld01)) ||
				((directie.at<float>(i, j) > ls02) && (directie.at<float>(i, j) <= ld02))) {

				if (nonMax.at<uchar>(i - 1, j) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i + 1, j) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

			if (((directie.at<float>(i, j) > ls31) && (directie.at<float>(i, j) <= ld31)) ||
				((directie.at<float>(i, j) > ls32) && (directie.at<float>(i, j) <= ld32))) {

				if (nonMax.at<uchar>(i - 1, j - 1) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i + 1, j + 1) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

			if (((directie.at<float>(i, j) > ls21) && (directie.at<float>(i, j) <= ld21)) ||
				((directie.at<float>(i, j) > ls22) && (directie.at<float>(i, j) <= ld22))) {

				if (nonMax.at<uchar>(i, j - 1) >= nonMax.at<uchar>(i, j) ||
					nonMax.at<uchar>(i, j + 1) >= nonMax.at<uchar>(i, j))
					nonMax.at<uchar>(i, j) = 0;
			}

		}
	}

	imshow("Non-max", nonMax);
	// Step 4: adaptive binarization
	Mat binarizare = nonMax.clone();

	int hist[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[binarizare.at<uchar>(i, j)]++;
		}
	}

	int nrPuncteZero = hist[0];
	int nrTotalPuncte = (width - 2) * (height - 2);
	float p = 0.15;
	int nrPuncteMuchie = p * (nrTotalPuncte - nrPuncteZero);
	int nrNonMuchie = (1 - p) * (nrTotalPuncte - nrPuncteZero);

	int treshold = 255;
	sum = 0;
	int x;
	for (x = 1; x < 256; x++) {
		sum += hist[x];
		if (sum > nrNonMuchie) {
			break;
		}
	}
	treshold = x;
	int pragAdaptiv = treshold;
	std::cout << pragAdaptiv << std::endl;
	std::cout << nrTotalPuncte << std::endl;
	std::cout << nrPuncteZero << std::endl;
	std::cout << nrNonMuchie << std::endl;
	std::cout << sum << std::endl;

	// Step 5: Extinderea muchiilor prin histereza
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (binarizare.at<uchar>(i, j) > pragAdaptiv) {
				binarizare.at<uchar>(i, j) = 255;
			}
			else {
				if (binarizare.at<uchar>(i, j) > 0.6 * pragAdaptiv) {
					binarizare.at<uchar>(i, j) = 128;
				}
				else {
					binarizare.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	std::queue<Point> coada;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (binarizare.at<uchar>(i, j) == 255) {
				coada.push(Point(i, j));
				while (!coada.empty()) {
					Point punct = coada.front();
					coada.pop();
					for (int k = -1; k <= 1; k++) {
						for (int l = -1; l <= 1; l++) {
							if (binarizare.at<uchar>(punct.x + k, punct.y + l) == 128) {
								binarizare.at<uchar>(punct.x + k, punct.y + l) = 255;
								coada.push(Point(punct.x + k, punct.y + l));
							}
						}
					}
				}
			}
		}
	}
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (binarizare.at<uchar>(i, j) == 128) {
				binarizare.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("Canny", binarizare);
	// Open the image to reduce the number of holes in the contour
	Mat temp = erode(binarizare.clone());
	Mat opened = dilate(temp);
	imshow("Open", opened);
	binarizare = opened.clone();

	// Step 6: Contour detection on the binary image & calculate leaf properties
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binarizare.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double minArea, maxArea, minCircularity, maxCircularity;
	calculateLeafProperties(binarizare, minArea, maxArea, minCircularity, maxCircularity, contours);

	std::cout << "Min Leaf Area: " << minArea << std::endl;
	std::cout << "Max Leaf Area: " << maxArea << std::endl;
	std::cout << "Min Circularity: " << minCircularity << std::endl;
	std::cout << "Max Circularity: " << maxCircularity << std::endl;

	/*cv::Mat maskedImage;
	binarizare.copyTo(maskedImage);
	cv::Rect stemROI = findStemCoordinates(maskedImage);
	maskedImage(stemROI) = 0;*/

	// Step 7: Filter contours based on area and shape
	std::vector<std::vector<cv::Point>> leafContours;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > maxArea/3 && area < maxArea*5) {
			double perimeter = cv::arcLength(contour, true);
			double circularity = 4 * CV_PI * area / (perimeter * perimeter);
			if (circularity > minCircularity && circularity < maxCircularity) {
				leafContours.push_back(contour);
			}
		}
	}


	// Only keep what's inside the contours
	cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
	//cv::findContours(binarizare.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::drawContours(mask, leafContours, -1, cv::Scalar(255), cv::FILLED);

	cv::Mat result2 = srcColor.clone();
	for (int y = 0; y < result2.rows; ++y) {
		for (int x = 0; x < result2.cols; ++x) {
			if (mask.at<uchar>(y, x) == 255) {
				result2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
			}
		}
	}

	// Step 8: Draw contours on the original image
	cv::Mat result = src.clone();
	cv::drawContours(result, leafContours, -1, cv::Scalar(0), 2);

	// Display the result
	cv::imshow("Leaves", result);
	cv::imshow("Original Image with Red Leaves", result2);
	cv::waitKey();
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Additive Gray Factor\n");
		printf(" 11 - Multiplicative Gray Factor\n");
		printf(" 12 - Create imagie\n");
		printf(" 13 - 3 Color channels to grayscale\n");
		printf(" 14 - Grayscale to Black-White\n");
		printf(" 15 - RGB to HSV\n");
		printf(" 16 - Is inside?\n");
		printf(" 17 - Histogram\n");
		printf(" 18 - FDP\n");
		printf(" 19 - Multiple tresholds\n");
		printf(" 20 - Floyd Steinberg\n");
		printf(" 21 - Proprerties\n");
		printf(" 22 - Width traversal\n");
		printf(" 23 - Two Traversals algorithm\n");
		printf(" 24 - Contour drawing\n");
		printf(" 25 - Reconstruct Contour\n");
		printf(" 26 - Dilate\n");
		printf(" 27 - Erode\n");
		printf(" 28 - Open\n");
		printf(" 29 - Close\n");
		printf(" 30 - Boundary Extraction\n");
		printf(" 31 - Region Filling\n");
		printf(" 32 - Cumulative Histogram\n");
		printf(" 33 - Binary Automated Treshold\n");
		printf(" 34 - Contrast Modification\n");
		printf(" 35 - Gamma Correction\n");
		printf(" 36 - Histogram Equalization\n");
		printf(" 37 - Low-Pass Filters\n");
		printf(" 38 - High-Pass Filters\n");
		printf(" 39 - Magnitude Calculation\n");
		printf(" 40 - Low-Pass Frequential Domain Filters\n");
		printf(" 41 - High-Pass Frequential Domain Filters\n");
		printf(" 42 - Salt and Pepper noise Filter (Median, Minimal, Maximal)\n");
		printf(" 43 - Gaussian noise Filter\n");
		printf(" 44 - Canny edge detection\n");
		printf(" 45 - Leaf detection\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			additiveGray();
			break;
		case 11:
			multiplicativeGray();
			break;
		case 12:
			creareImagine4col();
			break;
		case 13:
			colorChannels();
			break;
		case 14:
			blackWhite();
			break;
		case 15:
			testRGB2HSV();
			break;
		case 16:
			int i, j;
			std::cout << "i, j=";
			std::cin >> i >> j;
			isInside2(i, j);
			break;
		case 17:
			int* hist1;
			hist1 = testHistogram();
			for (int i = 0; i < 255; i++)
				printf("%d ", hist1[i]);
			break;
		case 18:
			char fname[MAX_PATH];
			while (openFileDlg(fname)) //tine fereastra de deschidere img deschisa 
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE); //citeste imaginea in grayscale
				int height = src.rows;
				int width = src.cols;
				float* fdp1;
				fdp1 = testFDP(src, width, height);
				for (int i = 0; i < 255; i++)
					printf("%f ", fdp1[i]);
				break;
			}
		case 19:
			multipleTresholds();
			break;
		case 20:
			floydSteinberg();
			break;
		case 21:
			onMouse();
			break;
		case 22:
			int vecini;
			std::cout << "Cu cati vecini sa fie facuta verificarea? (4 sau 8)" << std::endl;
			std::cin >> vecini;
			widthTraversal(vecini);
			break;
		case 23:
			int vecini2;
			std::cout << "Cu cati vecini sa fie facuta verificarea? (4 sau 8)" << std::endl;
			std::cin >> vecini2;
			secondTraversal(vecini2);
			break;
		case 24:
			contourCod();
			break;
		case 25:
			reconstructContour();
			break;
		case 26:
			int nd;
			std::cout << "De cate ori sa fie facuta dilatarea?" << std::endl;
			std::cin >> nd;
			dilateN(nd);
			break;
		case 27:
			int ne;
			std::cout << "De cate ori sa fie facuta eroziunea?" << std::endl;
			std::cin >> ne;
			erodeN(ne);
			break;
		case 28:
			int no;
			std::cout << "De cate ori sa fie facuta deschiderea?" << std::endl;
			std::cin >> no;
			openN(no);
			break;
		case 29:
			int nc;
			std::cout << "De cate ori sa fie facuta inchiderea?" << std::endl;
			std::cin >> nc;
			closeN(nc);
			break;
		case 30:
			contour();
			break;
		case 31:
			regionFilling();
			break;
		case 32:
			cumulativeHistogram();
			break;
		case 33:
			autoBinary();
			break;
		case 34:
			int gMin, gMax;
			std::cout << "gMin= " << std::endl;
			std::cin >> gMin;
			std::cout << "gMax= " << std::endl;
			std::cin >> gMax;
			contrastModification(gMin, gMax);
			break;
		case 35:
			float gamma;
			std::cout << "gamma= " << std::endl;
			std::cin >> gamma;
			gammaCorrection(gamma);
			break;
		case 36:
			histogramEqualization();
			break;
		case 37:
			int f1;
			std::cout << "Ce fel de filtru low pass? \n1. Medie artimetica\n2. Gaussian" << std::endl;
			std::cin >> f1;
			lowPassFilters(f1);
			break;
		case 38:
			int f2;
			std::cout << "Ce fel de filtru high pass? \n1. Laplace\n2. High-Pass" << std::endl;
			std::cin >> f2;
			highPassFilters(f2);
			break;
		case 39:
			magnitudeCalculator();
			break;
		case 40:
			int ff1;
			std::cout << "Ce fel de filtru low pass in domeniu frecvential?\n1. Ideal\n2. Gaussian" << std::endl;
			std::cin >> ff1;
			lowPassFrequential(ff1);
			break;
		case 41:
			int ff2;
			std::cout << "Ce fel de filtru gigh pass in domeniu frecvential?\n1. Ideal\n2. Gaussian" << std::endl;
			std::cin >> ff2;
			highPassFrequential(ff2);
			break;
		case 42:
			saltAndPepper();
			break;
		case 43:
			gaussianNoiseFilter();
			break;
		case 44:
			canny();
			break;
		case 45:
			detectLeaves();
			break;
		}
	} while (op != 0);
	return 0;
}