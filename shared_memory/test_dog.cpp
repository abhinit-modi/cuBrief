#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "CycleTimer.h"

//#define DEBUG

#ifdef DEBUG
/* When debugging is enabled, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__); 
#else
/* When debugging is disnabled, no code gets generated for these */
#define dbg_printf(...)
#endif


using namespace cv;
using namespace std;

extern void DoG_detector_serial
(Mat img, int **k_x, int **k_y, 
int *n, float th_c, float th_r,
int levels, float sigma);

extern void DoG_detector_cuda
(Mat img, int **k_x, int **k_y, 
int *n, float th_c, float th_r,
int levels, float sigma);

/* Scales to be tested */
int n_scales = 8;
string test_scales[] = {"32", "64", "128", "256", "512", "1024", "2048", "4096"};

/* Images sets */
int sets = 4;
string img_sets[] = {"grafiti", "boat","ubc","leuven"};

/* Benchmark directory */
string test_dir = "./benchmark/";
string format_img = ".png";
 
void drawKeyPoints(Mat image, int* x, int* y, int n, std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<n;i++){
        Point center = Point(*x, *y);

        cv::circle(target, center, 1, Scalar(255, 0, 0), 1);
        x++;
        y++;
    }

    imwrite(output_file, target);
}

void test_opencv_dog(Mat img)
{

}

void test_serial_dog(Mat img, bool print)
{
    int *kp_x, *kp_y, n = 0;

    DoG_detector_serial(img, &kp_x, &kp_y, &n, 7.0, 12.0, 6, 1.0);

    free(kp_x);
    free(kp_y);

    if (print)
        cout<<"Detected "<<n<<" keypoints"<<endl;

}

void test_cuda_dog(Mat img, bool print)
{
    int *kp_x_h, *kp_y_h, kp_h = 0;

    DoG_detector_cuda(img, &kp_x_h, &kp_y_h, &kp_h, 7.0, 12.0, 6, 1.0);

    free(kp_x_h);
    free(kp_y_h);

    if (print)
        cout<<"Detected "<<kp_h<<" keypoints"<<endl; 
}

void processUsingCuda(std::string input_file, std::string output_file)
{
    cv::Mat input = cv::imread( input_file, CV_LOAD_IMAGE_GRAYSCALE);

    Mat input_float;
    input.convertTo(input_float, CV_32F);

    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }
    int *kp_x_h, *kp_y_h, kp_h = 0;

    double start = CycleTimer::currentSeconds();

    DoG_detector_cuda(input_float, &kp_x_h, &kp_y_h, &kp_h, 7.0, 12.0, 6, 1.0);

    double end = CycleTimer::currentSeconds();

    cout << "took time: " << end-start << endl;
    drawKeyPoints(input, kp_x_h, kp_y_h, kp_h,"shit_man.jpg");
}

int main(int argc, char ** argv) 
{
    double start = 0.0;
    double end = 0.0;
    processUsingCuda(argv[1],argv[2]);
    processUsingCuda(argv[1],argv[2]);
    processUsingCuda(argv[1],argv[2]);
    processUsingCuda(argv[1],argv[2]);
    return 0;

    cv::Mat input = cv::imread("ferrari.png", CV_LOAD_IMAGE_GRAYSCALE);

    Mat input_float;
    input.convertTo(input_float, CV_32F);

    start = CycleTimer::currentSeconds();

    test_cuda_dog(input_float, false);

    end = CycleTimer::currentSeconds();

    for (int set = 0; set < sets; set++)
    {
        cout <<"Testing set: " << img_sets[set] <<endl;

        for (int s = 0; s < n_scales; s++)
        {
            /* Load test image */
            string img = test_dir + img_sets[set] + "/" + img_sets[set] 
                        + test_scales[s] + format_img;
            //cout<<"IMAGE is "<< img << endl; 
            input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);

            input_float;
            input.convertTo(input_float, CV_32F);

            start = CycleTimer::currentSeconds();

            test_cuda_dog(input_float, false);

            end = CycleTimer::currentSeconds();

            cout<<""<<end-start<<""<<endl;
        }
    }
    return 0;

}
