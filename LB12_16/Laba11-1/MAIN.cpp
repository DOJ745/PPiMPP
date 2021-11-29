﻿// Laba11-1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;

// CONSTANTS

Mat src_gray;
int thresh = 100;
RNG rng(12345);

void thresh_callback(int, void*);

void thresh_callback(int, void*)
{
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    cout << (int)contours.size() + "";
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
}

Mat imgCH;
int max_thresh = 255;
const char* source_window = "SOURCE image";
const char* corners_window = "CORNERS detected";
void cornerHarris_demo(int, void*);

void cornerHarris_demo(int, void*)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    // определение углов
    Mat dst = Mat::zeros(imgCH.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    // нормализация выходного вектора углов
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    // рисование кругов вокруг углов
    // параметр thresh определяет порог отсечения (0-255)
    // чем он меньше, тем больше точек будет отрисовано
    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    namedWindow(corners_window);
    imshow(corners_window, dst_norm_scaled);
}

static void LB12() 
{
    Mat srcImg = imread("beer.jpg", 1);

    float kernelMatrix[9];

    kernelMatrix[0] = -1;
    kernelMatrix[1] = -1;
    kernelMatrix[2] = -1;

    kernelMatrix[3] =-1;
    kernelMatrix[4] = 9;
    kernelMatrix[5] = -1;

    kernelMatrix[6] = -1;
    kernelMatrix[7] = -1;
    kernelMatrix[8] = -1;

    Mat newImg,resultImg;
    const Mat kernelMat(3, 3, CV_32FC1, (float*)kernelMatrix);
    filter2D(srcImg, newImg, -1, kernelMat);

    imshow("SOURCE", srcImg);
    imshow("FILTERED (kernel)", newImg);

    waitKey(0);

    Mat blurImg, boxFilterImg, GaussianBlurImg, medianBlurImg;
    Size ksize = Size(31,31);

    blur(srcImg, blurImg, ksize);

    boxFilter(srcImg, boxFilterImg, -1,
        ksize);

    GaussianBlur(srcImg, GaussianBlurImg, ksize,
        0, 0,
        BORDER_DEFAULT);

    medianBlur(srcImg, medianBlurImg, 31);

    imshow("BLUR", blurImg);
    imshow("BOX Filter", boxFilterImg);
    imshow("GAUSSIAN Blur", GaussianBlurImg);
    imshow("MEDIAN Blur", medianBlurImg);

    waitKey(0);

    Mat binaryImg;

    cvtColor(srcImg, binaryImg, COLOR_RGB2GRAY, 0);
    threshold(binaryImg, binaryImg, 120, 255, THRESH_BINARY_INV); 
        imshow("BINARY Image", binaryImg);

    Mat erodeImg, dilateImg, element, diffImg;
    element = Mat();

    erode(binaryImg, erodeImg, element);
    dilate(binaryImg, dilateImg, element);

    cv::absdiff(binaryImg, erodeImg, diffImg);

    imshow("Erode", erodeImg);
    imshow("Dilate", dilateImg);
    imshow("Difference", diffImg);

    waitKey();
}

static void LB13() 
{
    VideoCapture capture(CAP_ANY);
    if (!capture.isOpened())
    {
        cout << "ERROR: Can't initialize camera capture" << endl;
        return ;
    }
    else { cout << "Camera is found!"; }

    Mat frame;
    size_t nFrames = 0;
    int enableProcessing = 0;
    int64 t0 = cv::getTickCount();
    int64 processingTime = 0;

    for (;;)
    {
        capture >> frame; // read the next frame from camera
        if (frame.empty())
        {
            cout << "ERROR: Can't grab camera frame." << endl;
            break;
        }
        if (enableProcessing==0) { imshow("Frame", frame); }

        if (enableProcessing == 1)
        {
            Mat processed;
            Sobel(frame, processed, -1, 1, 0);

            imshow("Frame", processed);
        }

        if (enableProcessing == 2)
        {
            Mat processed;
            Laplacian(frame, processed, -1);

            imshow("Frame", processed);
        }

        if (enableProcessing == 3)
        {
            Mat processed;
            Laplacian(frame, processed, -1);

            Mat img, grayImg, edgesImg;
            double lowThreshold = 30, uppThreshold = 50;

            blur(frame, img, Size(3, 3));
            cvtColor(img, grayImg, COLOR_RGB2GRAY); 

            // применение детектора Кэнни
            Canny(grayImg, edgesImg, lowThreshold, uppThreshold);

            imshow("Frame", edgesImg);
        }
        int key = waitKey(1);
        if (key == 27/*ESC*/) break;
        if (key == 32/*SPACE*/)
        {
            enableProcessing++;
            if (enableProcessing == 4)
                enableProcessing = 0;
            cout << "Enable frame processing ('space' key): " << enableProcessing << endl;
        }
    }
    cout << "Number of captured frames: " << nFrames << endl;
}

static void LB14() 
{
    Mat img = imread("money.jpg", 1),contorImg;

    cvtColor(img, src_gray, COLOR_BGR2GRAY);

    blur(src_gray, src_gray, Size(9, 9));
    threshold(src_gray, src_gray, 250, 255, THRESH_BINARY_INV);

    erode(src_gray, src_gray, Mat(), Point(-1, -1), 10);
    imshow("Grey money", src_gray);

    waitKey();

    const char* source_window = "Source";

    namedWindow(source_window);
    imshow(source_window, img);

    const int max_thresh = 255;
    thresh_callback(0, 0);

    waitKey();

    img = imread("sydoku.jpg", 1);
    Mat dst, cdst, cdstP;

    Canny(img, dst, 50, 200, 3);
    cvtColor(dst, cdst, COLOR_GRAY2BGR);

    cdstP = cdst.clone();
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        pt1.x = cvRound( x0 + 1000 * (-b) );
        pt1.y = cvRound( y0 + 1000 * (a) );
        pt2.x = cvRound( x0 - 1000 * (-b) );
        pt2.y = cvRound( y0 - 1000 * (a) );

        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    // Probabilistic Line Transform
    
    // Show results
    imshow("Source", img);
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    //imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    waitKey();

    img = imread("circle.jpg", 1);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;

    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
        gray.rows / 16, // change this value to detect circles with
        //different distances to each other
        100, 30, 1, 30 // change the last two parameters
    ); // (min_radius & max_radius) to detect larger circles

    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(img, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
        // circle outline
        int radius = c[2]; 
        circle(img, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
    }

    const char* detected_circles = "Detected circles";
    namedWindow(detected_circles);
    imshow(detected_circles, img);

    waitKey();
}

static void LB15() 
{

    imgCH = imread("list.png", 1);
    cvtColor(imgCH, src_gray, COLOR_BGR2GRAY);

    namedWindow(source_window);
    thresh = 200;

    imshow(source_window, imgCH);
    cornerHarris_demo(0, 0);

    waitKey();

    Mat img = imread("list.png", 1),corners;
    RNG rng(12345);
    cvtColor(img, src_gray, COLOR_BGR2GRAY);

    namedWindow(source_window);
    imshow(source_window, img);

    goodFeaturesToTrack(src_gray,corners, 50, 0.01, 10, Mat(), 3, false, 0.04); 
    cout << "** Number of corners detected: " << corners.size() << endl;

    int radius = 4;

    /*for (size_t i = 0; i < corners.size(); i++)
    {
        circle(copy, corners[i], radius, Scalar(rng.uniform(0, 255),
            rng.uniform(0, 256), rng.uniform(0, 256)), FILLED);
    }
    namedWindow(source_window);

    imshow(source_window, copy);*/
     
    waitKey();
}

static void LB16() 
{

}

int main()
{
    int input;
    cout << "LB12 - LB16\n";

    cout << "1. LB12\n";
    cout << "2. LB13\n";
    cout << "3. LB14\n";
    cout << "4. LB15\n";
    cout << "5. LB16\n";
    cout << "0. Exit\n";
    cout << "Selection: ";
    cin >> input;

    switch (input) {

    case 1: LB12(); break;
    case 2: LB13(); break;
    case 3: LB14(); break;
    case 4: LB15(); break;
    case 5: LB15(); break;
    case 0: cout << "Exiting...\n"; break;

    default:
        cout << "Error, bad input, quitting...\n";
        break;
    }
    cin.get();
}

