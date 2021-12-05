#include <opencv2/core.hpp>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//#include <opencv2/optflow/motempl.hpp>

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
    Canny(SRC_GRAY, canny_input, Thresh, (double)Thresh * (double)2);

    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(canny_input, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

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

CascadeClassifier faceCascade, eyesCascade;

string faceCascadeName = "frontal_face2.xml";
string eyesCascadeName = "eyes2.xml";

void detectAndDisplay(Mat frame);
void detectAndDisplay(Mat frame)
{

    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    vector<Rect> faces;
    faceCascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

        ellipse(frame, 
            center, 
            Size(faces[i].width / 2, faces[i].height / 2), 
            0, 0, 360, Scalar(0, 0, 255), 4);

        Mat faceROI = frame_gray(faces[i]);

        vector<Rect> eyes;
        eyesCascade.detectMultiScale(faceROI, eyes);

        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(
                faces[i].x + eyes[j].x + eyes[j].width / 2, 
                faces[i].y + eyes[j].y + eyes[j].height / 2);

            int radius = cvRound(
                ((double)eyes[j].width + (double)eyes[j].height) * 0.25
            );

            circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
        }

    }

    imshow("Capture - Face detection", frame);
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
    Mat img = imread("star.png", 1), contorImg;

    cvtColor(img, SRC_GRAY, COLOR_BGR2GRAY);

    blur(SRC_GRAY, SRC_GRAY, Size(9, 9));
    threshold(SRC_GRAY, SRC_GRAY, 200, 255, THRESH_BINARY_INV);
    
    namedWindow("GRAY IMAGE");
    imshow("GRAY IMAGE", SRC_GRAY);

    waitKey();


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
    faceCascade.load(faceCascadeName);
    eyesCascade.load(eyesCascadeName);

    VideoCapture capture(0);

    if (!capture.isOpened()) {
        cerr << "Unable to open: " << endl;
    }
    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        detectAndDisplay(frame);

        if (waitKey(10) == 27) { break; }
    }
}

static void LB17_1()
{
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;

    pBackSub = createBackgroundSubtractorMOG2();
    //pBackSub = createBackgroundSubtractorKNN();

    VideoCapture capture(0);

    if (!capture.isOpened()) {
        cerr << "Unable to open: " << endl;
    }

    Mat frame, fgMask;
    while (true) 
    {
        capture >> frame;
        if (frame.empty())
            break;

        //update the background model
        pBackSub->apply(frame, fgMask);
        //get the frame number and write it on the current frame
        rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
            cv::Scalar(255, 255, 255), -1);
        stringstream ss;

        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();

        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);
        //get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }

}

static void LB17_2() 
{
    VideoCapture capture(0);

    if (!capture.isOpened()) {
        cerr << "Unable to open: " << endl;
    }
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;

    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
    }

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    // Take first frame and find corners in it
    capture >> old_frame;

    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    while (true) 
    {

        Mat frame, frame_gray;
        capture >> frame;

        if (frame.empty())
            break;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // calculate optical flow
        vector<uchar> status;
        vector<float> err;

        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);

        vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        add(frame, mask, img);
        imshow("Frame", img);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
}

static void LB17_3()
{

}

int main()
{
    //LB12();
    //LB13();
    //LB14();
    //LB15();
    //LB16();

    LB17_1();
    //LB17_2();
    //LB17_3();
}


