#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/mat.hpp>

#include <iostream>
using namespace cv;
using namespace std;

// CONSTANTS

Mat SRC_GRAY;
int Thresh = 100;
RNG RandomNumGen(12345);

void draw_contures(int, void*);
void draw_contures(int, void*)
{
    Mat canny_input;
    Canny(SRC_GRAY, canny_input, Thresh, Thresh * 2);

    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(canny_input, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros(canny_input.size(), CV_8UC3);
    cout << "Contours size - " << (int)contours.size();

    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(
            RandomNumGen.uniform(0, 256), 
            RandomNumGen.uniform(0, 256), 
            RandomNumGen.uniform(0, 256)
        );

        drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    imshow("CONTOROUS", drawing);
}

Mat IMG_CH;
int max_thresh = 255;

const char* SOURCE_WINDOW = "SOURCE image";
const char* corners_window = "HARRIS FUNC CORNERS detected";

void cornerHarris_demo(int, void*);
void cornerHarris_demo(int, void*)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    Mat dst = Mat::zeros(IMG_CH.size(), CV_32FC1);
    cornerHarris(SRC_GRAY, dst, blockSize, apertureSize, k);

    Mat dst_norm, dst_norm_scaled;

    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > Thresh)
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

    kernelMatrix[3] = -1;
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
        capture >> frame;
        if (frame.empty())
        {
            cout << "ERROR: Can't grab camera frame." << endl;
            break;
        }
        if (enableProcessing == 0) { imshow("Frame", frame); }

        if (enableProcessing == 1)
        {
            Mat processed;
            Sobel(frame, processed, -1, 1, 0);

            imshow("SOBEL Frame", processed);
        }

        if (enableProcessing == 2)
        {
            Mat processed;
            Laplacian(frame, processed, -1);

            imshow("LAPLACIAN Frame", processed);
        }

        if (enableProcessing == 3)
        {
            Mat processed;
            Laplacian(frame, processed, -1);

            Mat img, grayImg, edgesImg;
            double lowThreshold = 30, uppThreshold = 50;

            blur(frame, img, Size(3, 3));
            cvtColor(img, grayImg, COLOR_RGB2GRAY); 

            Canny(grayImg, edgesImg, lowThreshold, uppThreshold);

            imshow("EDGES Frame", edgesImg);
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
    Mat img = imread("stars.png", 1), contorImg;

    cvtColor(img, SRC_GRAY, COLOR_BGR2GRAY);

    blur(SRC_GRAY, SRC_GRAY, Size(9, 9));
    threshold(SRC_GRAY, SRC_GRAY, 195, 255, THRESH_BINARY_INV);

    erode(SRC_GRAY, SRC_GRAY, Mat(), Point(-1, -1), 10);

    namedWindow(SOURCE_WINDOW);
    imshow(SOURCE_WINDOW, img);

    const int max_thresh = 255;
    draw_contures(0, 0);

    waitKey();

    img = imread("nonogram.png", 1);
    Mat dst, cdst, cdstP;

    Canny(img, dst, 50, 200, 3);
    cvtColor(dst, cdst, COLOR_GRAY2BGR);

    cdstP = cdst.clone();
    vector<Vec2f> lines;

    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

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
    
    imshow(SOURCE_WINDOW, img);
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in black) - Probabilistic Line Transform", cdstP);

    waitKey();

    img = imread("circles.png", 1);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;

    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
        gray.rows / 16, // change this value to detect circles with
        100, 36, 1, 160
    );

    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);

        circle(img, center, 1, Scalar(0, 0, 0), 3, LINE_AA);

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

    IMG_CH = imread("list.png", 1);
    cvtColor(IMG_CH, SRC_GRAY, COLOR_BGR2GRAY);

    namedWindow(SOURCE_WINDOW);
    Thresh = 200;

    imshow(SOURCE_WINDOW, IMG_CH);
    cornerHarris_demo(0, 0);

    waitKey();

    Mat img = imread("list.png", 1);

    vector<Point2f>  corners;
    RNG RandomNumGen(12345);

    cvtColor(img, SRC_GRAY, COLOR_BGR2GRAY);

    namedWindow(SOURCE_WINDOW);
    imshow(SOURCE_WINDOW, img);

    goodFeaturesToTrack(SRC_GRAY, corners, 50, 0.01, 10, Mat(), 3, false, 0.04); 
    cout << "Number of corners detected: " << corners.size() << endl;

    int radius = 4;

    for (size_t i = 0; i < corners.size(); i++)
    {
        circle(IMG_CH, corners[i], radius, Scalar(RandomNumGen.uniform(0, 256),
            RandomNumGen.uniform(0, 256), RandomNumGen.uniform(0, 256)), FILLED);
    }

    namedWindow(SOURCE_WINDOW);
    imshow(SOURCE_WINDOW, IMG_CH);
     
    waitKey();

    Point2f srcTri[3];

    srcTri[0] = Point2f(0.f, 0.f);
    srcTri[1] = Point2f(IMG_CH.cols - 1.f, 0.f);
    srcTri[2] = Point2f(0.f, IMG_CH.rows - 1.f);

    Point2f dstTri[3];

    dstTri[0] = Point2f(0.f, IMG_CH.rows * 0.33f);
    dstTri[1] = Point2f(IMG_CH.cols * 0.85f, IMG_CH.rows * 0.25f);
    dstTri[2] = Point2f(IMG_CH.cols * 0.15f, IMG_CH.rows * 0.7f);

    Mat warp_mat = getAffineTransform(srcTri, dstTri);
    Mat warp_dst = Mat::zeros(IMG_CH.rows, IMG_CH.cols, IMG_CH.type());

    warpAffine(IMG_CH, warp_dst, warp_mat, warp_dst.size());

    Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
    double angle = 90.0;
    double scale = 1;
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
    Mat warp_rotate_dst;

    warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());

    imshow("Warp", warp_dst);
    imshow("Warp ROTATE", warp_rotate_dst);

    waitKey();
}

static void LB16() 
{

}

int main()
{
    //LB12();
    //LB13();
    //LB14();
    LB15();
    //LB16();
}


