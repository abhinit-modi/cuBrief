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
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "CycleTimer.h"
#include "filter.h"
#include "filter.cu_incl"

#define DEBUG

#ifdef DEBUG
/* When debugging is enabled, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__); 
#else
/* When debugging is disnabled, no code gets generated for these */
#define dbg_printf(...)
#endif


using namespace cv;
using namespace cv::cuda;
using namespace std;

#define MAX_IMG_SZ 4096 * 4096 *2

#define MAX_PYRAMIDS 5

/* Global memory for keypoints */
const int max_kp = 12000;

__device__ int kp = 0;
__device__ int kp_x[max_kp];
__device__ int kp_y[max_kp];

__device__ float d[MAX_PYRAMIDS][MAX_IMG_SZ];
__device__ float s[MAX_PYRAMIDS + 1][MAX_IMG_SZ];

__device__ int ref_b[256] = {52, 70, 37, 11, 6, 65, 75, 10, 25, 
38, 48, 75, 73, 7, 37, 3, 16, 35, 44, 
50, 56, 52, 65, 52, 48, 48, 77, 45, 48, 
66, 61, 63, 14, 27, 17, 56, 55, 38, 69, 
2, 51, 12, 6, 31, 7, 60, 19, 3, 40, 
3, 34, 28, 23, 4, 61, 30, 49, 31, 56, 
65, 26, 18, 79, 49, 5, 50, 19, 67, 72, 
75, 15, 3, 40, 54, 33, 52, 16, 79, 8, 
45, 25, 42, 64, 77, 63, 47, 26, 11, 56, 
23, 74, 19, 33, 63, 57, 52, 46, 14, 48, 
24, 59, 56, 27, 70, 38, 63, 44, 52, 51, 
33, 9, 71, 2, 66, 81, 36, 43, 20, 4, 
76, 60, 55, 80, 54, 55, 54, 18, 41, 68, 
10, 30, 57, 6, 53, 54, 35, 53, 80, 32, 
19, 27, 3, 47, 46, 58, 15, 32, 8, 67, 
50, 23, 67, 30, 29, 18, 53, 70, 67, 24, 
48, 19, 13, 16, 55, 22, 49, 17, 16, 7, 
67, 61, 68, 5, 17, 26, 66, 27, 52, 62, 
46, 24, 27, 57, 78, 23, 78, 71, 79, 47, 
72, 8, 10, 20, 43, 15, 69, 18, 81, 14, 
68, 38, 34, 62, 15, 5, 58, 27, 39, 72, 
61, 8, 1, 63, 21, 33, 18, 57, 11, 4, 
57, 58, 50, 61, 66, 65, 12, 29, 41, 3, 
34, 11, 59, 47, 71, 50, 63, 38, 32, 35, 
11, 54, 77, 21, 38, 4, 7, 21, 57, 17, 
44, 12, 22, 34, 31, 41, 64};

__device__ int ref_a[256] = {75, 70, 47, 48, 70, 1, 3, 38, 20, 
39, 16, 53, 28, 71, 31, 15, 5, 34, 72, 
74, 53, 4, 57, 15, 59, 16, 51, 79, 9, 
7, 77, 69, 32, 73, 3, 28, 42, 31, 22, 
4, 63, 1, 1, 8, 16, 17, 63, 80, 36, 
48, 16, 74, 54, 37, 33, 65, 7, 12, 24, 
52, 72, 38, 34, 70, 54, 77, 54, 73, 44, 
81, 7, 69, 80, 60, 38, 28, 2, 79, 51, 
47, 17, 8, 7, 44, 44, 25, 75, 34, 48, 
15, 30, 54, 6, 28, 33, 2, 13, 54, 47, 
42, 31, 29, 46, 65, 18, 39, 64, 72, 41, 
19, 64, 14, 15, 66, 54, 42, 70, 37, 32, 
51, 45, 70, 9, 20, 67, 64, 20, 4, 50, 
9, 16, 81, 27, 57, 48, 37, 72, 27, 21, 
78, 73, 44, 60, 76, 31, 54, 33, 56, 10, 
55, 52, 24, 56, 56, 79, 45, 4, 38, 65, 
48, 38, 62, 19, 26, 57, 61, 19, 47, 56, 
26, 39, 79, 76, 25, 44, 81, 22, 19, 55, 
24, 17, 43, 78, 39, 66, 59, 14, 11, 24, 
57, 16, 12, 53, 26, 33, 53, 34, 45, 63, 
69, 21, 32, 58, 73, 49, 62, 2, 73, 19, 
72, 81, 13, 13, 67, 72, 11, 69, 61, 47, 
24, 79, 22, 3, 62, 67, 53, 14, 80, 20, 
72, 17, 10, 10, 45, 81, 57, 33, 61, 16, 
16, 9, 66, 3, 63, 36, 63, 67, 41, 30, 
64, 35, 49, 47, 22, 44, 40 };

__device__ inline void smoothedSum(float* src, short2 pt, int step, uchar* temp, int idx)
{
    const int img_y = (int)pt.y - 4;
    const int img_x = (int)pt.x - 4;
    uchar* t = temp + 32*idx;

    for(int j=0;j<32;j++){
        uchar dig = '\0';
        for(int i=0;i<8;i++){
            int index = j*8 + i;
            int start1_x = img_x + (ref_a[index]-1)%9;
            int start1_y = img_y + (ref_a[index]-1)/9;

            int start2_x = img_x + (ref_b[index]-1)%9;
            int start2_y = img_y + (ref_b[index]-1)/9;
            
            int result = src[start1_y*step + start1_x] < src[start2_y*step + start2_x];
            dig = dig | (uchar(result));
            if(i!=7)
                dig = dig<<1;
        }

        // if(idx == 6){
        //     printf("result is %d\n", dig);
        // }
        t[j] = dig;
    }
}

__device__ void img_to_s(float *img, float *s, int w, int h, int x, int y)
{
    for (int yy = -1; yy <= 1; yy++)
        for (int xx = -1; xx <= 1; xx++)
        {
            int x_i = x +xx;
            int y_i = y + yy;

            if (x_i < 0 || y_i < 0 || x_i >= w || y_i >= h)
                s[ (yy + 1) * 3 + (xx +1)] = 0;
            else
                s[ (yy + 1) * 3 + (xx +1)] = img[(y_i * w) + x_i];
        }

}

__global__ void vertical_blur( float* img,
                            int w,
                            int h,
                            filter_bank fb)
{

    /*2D Index of current thread */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int center = 3*(1) + 1;
    const int levels = fb.n;
    float fy[50];
    float fx[50];

    /* Out of bounds pixel */
    if (x >= w || y >= h)
        return;

    for (int i = 0; i < levels; i ++)
    {
        int k = fb.bank[i].k;

        linear_filter lf;
        lf.hy = fy;
        lf.hx = fx;
        lf.k = fb.bank[i].k;

        square_to_linear(fb.bank[i], &lf);
       
        s[i][(y*w) + x] = get_filter_response_vertical(img, w, h, lf, x, y);
    }
    //printf("Test1 \n");
}


__global__ void horizontal_blur( float* img,
                            int w,
                            int h,
                            filter_bank fb)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int center = 3*(1) + 1;
    const int levels = fb.n - 1;
    float fx[50];
    float fy[50];

    float sk, sk_1;

    if (x >= w || y >= h)
        return;


    int k = fb.bank[0].k;

    linear_filter lf;
    lf.hx = fx;
    lf.hy = fy;

    lf.k = fb.bank[0].k;

    sk = get_filter_response_horizontal(s[0], w, h, lf, x, y);
    

    for (int i = 0; i < levels; i ++)
    {
        k = fb.bank[i+1].k;

        linear_filter lf;
        lf.hx = fx;
        lf.hy = fy;
    
        lf.k = k;
        
        square_to_linear(fb.bank[i+1], &lf);

        sk_1 = get_filter_response_horizontal(s[i+1], w, h, lf, x, y);

        d[i][(y*w) + x] = sk - sk_1;

        sk = sk_1;
    }
    //printf("Test 2\n");
}


__global__ void DoG_Pyramid( float* img,
                            int w,
                            int h,
                            filter_bank fb)
{

    /*2D Index of current thread */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int center = 3*(1) + 1;
    const int levels = fb.n - 1;
    float sk, sk_1;

    /* Out of bounds pixel */
    if (x >= w || y >= h)
        return;

    sk = get_filter_response(img, w, h, fb.bank[0], x, y);

    for (int i = 0; i < levels; i ++)
    {
        //sk = get_filter_response(img, w, h, fb.bank[i], x, y);
        sk_1 = get_filter_response(img, w, h, fb.bank[i+1], x, y);        
        d[i][(y*w) + x] = sk - sk_1; 
        sk = sk_1;
    }

}
__global__ void DoG_Kernel( float* img,
                            int w,
                            int h,
                            filter_bank fb,
                            uchar *output)
{
    /*2D Index of current thread */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int center = 3*(1) + 1;
    const int levels = fb.n - 1;
    const float th_c = 7.0;
    const float th_r = 12.0;
    bool keypoint_detected = false;

    /* Out of bounds pixel */
    if (x >= w || y >= h)
        return;

    /* Pixels in the border */
    if ( (x >= w-4) || (y >= h-4) || x < 4 || y < 4)
        return;
 
    /* DoG first levels */

    /* D(k-1), D(k), D(k+1) */
    float d_1k[9], dk[9], dk_1[9];
  
    /* Regsiters to calculate Hessian of DoG */
    float dh[25], sh_1[25], sh[25];
 
    /* Compute D(k) and D(k+1) for first level */
    int idx;
    for (int i = 0; i < levels; i++)
    {        
        float current = d[i][(y*w) + x];
        bool ismax = true;
        bool ismin = true;

     /* If threshold test fails go to next iteration */
        if (fabs(current) < th_c)
            continue;
 
        img_to_s(d[i], dk, w, h, x, y);

        /* Current layer */
        ismax = ismax && is_max(dk, current);
        ismin = ismin && is_min(dk, current);

        if (!ismax && !ismin)
            continue;

        /* Layer below */
        if (i != levels - 1)
        {
            img_to_s(d[i+1], dk_1, w, h, x, y);

            ismax = ismax && is_max(dk_1, current);
            ismin = ismin && is_min(dk_1, current);

            if (!ismax && !ismin)
                continue;
        }

        /* Layer above */
        if (i != 0)
        {
            img_to_s(d[i-1], d_1k, w, h, x, y);
            ismax = ismax && is_max(d_1k, current);
            ismin = ismin && is_min(d_1k, current);

            if (!ismax && !ismin)
                continue;
        }


        float R = get_curvature(d[i], w, h, x, y);
    
        if (R > th_r)
            break;
        /* Atomically increase the number of keypoints
           and add the new found keypoint 
        */

        idx = atomicAdd(&kp, 1);

        kp_x[idx] = x;
        kp_y[idx] = y;
  
       // dbg_printf("Keypoint detected at x = %d, y= %d. idx is %d ," 
         //           "and level is %d, and intensity is %f\n", x, y, idx, i, dk[center]);
        keypoint_detected = true;
        break;
    }

    // if(keypoint_detected == true){
    //     short2 pt;
    //     pt.x = x;
    //     pt.y = y;
    //     smoothedSum(img, pt, w, output, idx);
    // }
}

uchar* DoG_detector_cuda(Mat img, int **k_x, int **k_y, int *n, float th_c, float th_r, 
                  int levels, float sigma)
{
    
    double start, end;

    /* Device image */
    float *gpu_img;
    float *img_ptr = (float*) img.ptr<float>();

    /* Get width and height */
    int w = img.cols;
    int h = img.rows;
   
    /* BLock width */
    int block_width = 32;
 
    /* Calculate image size in bytes */
    size_t img_sz = w * h * sizeof(float);
 
    /* Generate DoG Levels */
    float sigma_l[10];

    for (int i = -1; i < levels - 1; i ++) 
        sigma_l[i+1] = (float) i;

    /* Create device and host filter banks */
    filter_bank fb, fbd;
    create_DoG_bank (&fb, levels, sqrt(2), sigma, sigma_l);

    /* Copy device filter bank to host */
    copy_DoG_bank_device(&fb, &fbd);

    /* Allocate image memory in device */
    cudaMalloc(&gpu_img, img_sz);
        
    /* Copy image from host to device */
    cudaMemcpy(gpu_img, img_ptr, img_sz, cudaMemcpyHostToDevice);

    /* Calculate Grid Size */   
    const dim3 block(block_width, block_width);
    const dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    start = CycleTimer::currentSeconds();

    /* Launch Kernel */
    //DoG_Kernel<<<grid,block>>>(gpu_img, w, h, fbd);

    DoG_Pyramid<<<grid,block>>>(gpu_img, w, h, fbd);
    //vertical_blur<<<grid,block>>>(gpu_img, w, h, fbd);
    cudaDeviceSynchronize();
    end = CycleTimer::currentSeconds();
    cout<<"Filters took "<<end-start<<" seconds"<<endl;

    //horizontal_blur<<<grid,block>>>(gpu_img, w, h, fbd);
    //cudaDeviceSynchronize();
    //end = CycleTimer::currentSeconds();
    //cout<<"Filters took "<<end-start<<" seconds"<<endl;

    uchar *d_descriptor, *h_descriptor;
    cudaMalloc(&d_descriptor, 32*sizeof(uchar)*500);
    h_descriptor = (uchar*)malloc(32*sizeof(uchar)*500);

    start = CycleTimer::currentSeconds();
    DoG_Kernel<<<grid,block>>>(gpu_img, w, h, fbd, d_descriptor);
    cudaDeviceSynchronize();
    end = CycleTimer::currentSeconds();

    //cout<<"CUDA MINMAX kernel took "<<end-start<<" seconds"<<endl;

    //cout<<"CUDA KERNEL took "<<end-start<<" seconds"<<endl;

    // dbg_printf("Finished calling kernel\n");

    /* Free device memory */
    cudaFree(gpu_img);
    
    /* Copy results from device to host */

    cudaMemcpyFromSymbol(n, kp, sizeof(int));
    dbg_printf("Detected %d keypoints \n",*n);

    *k_y = (int *) malloc(*n * sizeof(int));
    *k_x = (int *) malloc(*n * sizeof(int));
    
    cudaMemcpyFromSymbol(*k_x, kp_x, sizeof(int)* (*n));
    cudaMemcpyFromSymbol(*k_y, kp_y, sizeof(int)* (*n));
    // start = CycleTimer::currentSeconds();
    cudaMemcpy(h_descriptor, d_descriptor, 32*sizeof(uchar)*(*n), cudaMemcpyDeviceToHost);
    // end = CycleTimer::currentSeconds();
    // cout<<"memcopy of descriptor took "<<end-start<<" seconds"<<endl;

    /* Clear kp */
    int zero = 0;
    cudaMemcpyToSymbol(kp, &zero, sizeof(int));


    cudaFree(d_descriptor);
    return h_descriptor;

}


