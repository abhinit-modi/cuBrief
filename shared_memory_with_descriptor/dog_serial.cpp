#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h> 

using namespace cv;
using namespace std;

#define MAX_KERNEL_LENGTH 10
#define NUM_FILTERS 6
#define MAX_KEYPOINTS 9000

const float pi = 3.141592;
const float e = 2.71828;
int total = 0;
Point keypoints[100000];

float get_ix_serial(Mat reference, int x, int y)
{
    float i_right = 0.0;
    float i_left = 0.0;

    if ( (x-1) >= 0)
       i_left = reference.at<float>(y,x-1);
     
    if ( (x+1) < reference.cols)
       i_right = reference.at<float>(y,x+1);

    return (i_right - i_left) / 2.0;

}

float get_iy_serial(Mat reference, int x, int y)
{
    float i_up = 0.0;
    float i_down = 0.0;

    if ( (y-1) >= 0)
       i_up =  reference.at<float>(y-1,x);

    if ( (y+1) < reference.rows)
       i_down = reference.at<float>(y+1,x);

    return (i_up - i_down) / 2.0;

}

float get_curvature_serial(Mat reference, int x, int y)
{


    float dx_left = get_ix_serial(reference,x-1,y);
    float dx_right = get_ix_serial(reference,x+1,y);
    float dx_up = get_ix_serial(reference,x,y-1);
    float dx_down = get_ix_serial(reference,x,y+1);

    float dy_up = get_iy_serial(reference,x,y-1);
    float dy_down = get_iy_serial(reference,x,y+1);

    float d_xx = (dx_right - dx_left) / 2.0;
    float d_yy = (dy_up - dy_down) / 2.0;
    float d_xy = (dx_up - dx_down) / 2.0;

    /* Using Young's theorem Dxy = Dyx */
    float d_yx = d_xy; 
    
    /* Hessian trace */
    float trace_h = d_xx + d_yy;
    
    /* Hessian determinant */
    float det_h = (d_xx * d_yy) - (d_xy * d_yx);
    
    /* Compute curature R */
    float R = (trace_h * trace_h) / det_h;

    return R;

}

bool check_for_max_serial(float value, Mat reference, int ic, int jc)
{   
   for(int ii=-1;ii<=1;ii++)
   {
        for(int jj=-1;jj<=1;jj++)
        {
            float base = 0;
            if(jc+ii >= 0 && jc+ii<reference.cols && ic+jj >=0 && ic+jj < reference.rows)
                base = reference.at<float>(ic+jj, jc+ii);
                
            if(value < base)
                return false; 
        }
    }

    return true;
}


bool check_for_min_serial(float value, Mat reference, int ic, int jc)
{
    for(int ii=-1;ii<=1;ii++)
    {
        for(int jj=-1;jj<=1;jj++)
        {
            float base = 0;

            if(jc+ii >= 0 && jc+ii<reference.cols && ic+jj >=0 && ic+jj < reference.rows)
                base = reference.at<float>(ic+jj, jc+ii);
                
            if(value > base)
                return false;
        }
    }

    return true;
}

void dump_image_serial(Mat reference)
{

    for (int i = 0; i < reference.rows; i++)
     { 
        cout<<"ROW: "<<i<<endl;
        for (int j = 0; j < reference.cols; j++)
         {
            cout<<reference.at<float>(i,j)<<endl;        
         }
     }
}

float *get_gaussian_kernel_serial(float sigma, int k_size)
{
    int radius = k_size / 2;
    float k = 1/(2 * pi * sigma * sigma);
    
    /* Allocate memory for the filter */
    float *filter = (float *) malloc(k_size*k_size*sizeof(float));

    float den = 2 * (sigma * sigma);
    
    float sum = 0.0;

    for (int y = radius*-1; y <= radius; y++)
    {
        for (int x = radius*-1; x <= radius; x++)
        {
            float num = (x*x + y*y);
            float res = k * exp((-1*num)/den);

            filter[k_size * (y + radius) + (x + radius) ] = res;
            sum+= res;
        }
    }

    for (int y = radius*-1; y <= radius; y++)
        for (int x = radius*-1; x <= radius; x++)
            if (sum != 0.0)
            filter[k_size * (y + radius) + (x + radius)]/= sum;

    return filter;
}

void dump_filter_serial(float *filter, int k)
{
    for (int i = 0; i < k; i++)
    {
        for(int j = 0; j < k; j++)
            cout<<filter[k*i + j]<<" ";
        cout<<endl;
    }

}

void DoG_detector_serial(Mat img, int **k_x, int **k_y, int *n, float th_c, float th_r,
                  int levels, float sigma)
{

    Mat stack[levels];
    Mat difference[levels-1];

    float sigma_l[levels]; 
    int kx_buffer[MAX_KEYPOINTS];
    int ky_buffer[MAX_KEYPOINTS];
    int detected_kp = 0;


    /* Generate levels parameters according to
      the number of levels
    */
    for (int i = -1; i < levels - 1; i ++)
        sigma_l[i+1] = (float) i;
      
    /* Generate Gaussian Pyramid */    
    for(int i=0; i < levels;i++)
    {
        float sig = pow(sqrt(2), sigma_l[i]);
        int ksize = floor(6*sig) + 1;

        GaussianBlur(img, stack[i], Size(ksize, ksize),  sig); 
    }    
   
    /* Generate DoG Pyramid using Gaussian Pyramid */
    for(int i=0;i<NUM_FILTERS-1;i++)
        difference[i] = -1 * (stack[i+1] - stack[i]);

    /*Find keypoints using the following tests:
        1. Threshold test
        2. Min Max test
        3. Curvature test
    */
    for(int index=0;index<NUM_FILTERS-1;index++)
    {
        Mat reference = difference[index];
       
        for(int ic=0;ic<img.rows;ic++)
        {
            for(int jc=0;jc<img.cols;jc++)
            {

                /* Pixels in the border */
                if ( (jc >= img.cols-4) || (ic >= img.rows-4) || jc < 4 || ic < 4)
                    continue;

                float value = reference.at<float>(ic,jc);
            
                bool is_max = true;
                bool is_min = true;
                
                /* Threshold test */
                if (fabs(value) < th_c)
                    continue;
 
                /* Current scale */ 
                is_max = check_for_max_serial(value, difference[index], ic, jc);
                is_min = check_for_min_serial(value, difference[index],ic,jc);

                if(!is_min && !is_max)continue;
                
                /* Scale below */

                if(index!=0)
                {
                    is_max = check_for_max_serial(value, difference[index-1], ic, jc) && is_max;
                    is_min = check_for_min_serial(value, difference[index-1],ic,jc) && is_min;

                    if(!is_min && !is_max)continue;
                }

                /* Scale above */
                if(index!=NUM_FILTERS-2)
                {
                    is_max = check_for_max_serial(value, difference[index+1], ic, jc) && is_max;
                    is_min = check_for_min_serial(value, difference[index+1],ic,jc) && is_min;
                    if(!is_min && !is_max)continue;
                }
                
                /* Curvature test */
                float curvature = get_curvature_serial(reference, jc, ic);

                if (curvature > th_r)
                    continue;

                if((is_min || is_max ))
                { 
                    kx_buffer[detected_kp] = jc;
                    ky_buffer[detected_kp] = ic;
                    detected_kp ++;
                }
            }
        }

        /*Allocate memory for output parameters and copy results */
        *k_x = (int *) malloc(detected_kp * sizeof(int));
        *k_y = (int *) malloc(detected_kp * sizeof(int));

        memcpy(*k_x, kx_buffer, detected_kp * sizeof(int));
        memcpy(*k_y, ky_buffer, detected_kp * sizeof(int));

        *n = detected_kp;
    }
}


