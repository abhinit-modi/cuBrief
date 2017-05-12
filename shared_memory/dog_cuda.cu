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

//#define DEBUG

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

#define BSZ 32 

/* Global memory for keypoints */
const int max_kp = 12000;

__device__ int kp = 0;
__device__ int kp_x[max_kp];
__device__ int kp_y[max_kp];

__device__ float d[MAX_PYRAMIDS][MAX_IMG_SZ];
__device__ float s[MAX_PYRAMIDS + 1][MAX_IMG_SZ];

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
    const int padding = fb.bank[fb.n - 1].k ;
    const int cache_width = blockDim.x + padding;
    const int cache_height = blockDim.y;

    __shared__ float cache_image[(BSZ+25)*BSZ+25];

    const int center = 3*(1) + 1;
    const int levels = fb.n;
    __shared__ float fy[50];
    __shared__ float fx[50];

    /* Out of bounds pixel */
//    if (x != 0 || y != 0)
    if (x >= w || y >= h)
        return;
      
    /* Collect shared cache_image */
    cache_image[(threadIdx.y * cache_width) + threadIdx.x + padding/2 ] = img[(y*w) + x];

    /* Borders are collected by first threads */
    if (threadIdx.x < padding/2 && (x- padding/2)  >= 0)
        cache_image[(threadIdx.y * cache_width) + threadIdx.x] = img[(y*w) + x - padding/2];

    /* And last threads */
    if (threadIdx.x > cache_width - padding/2 && (x + padding/2) < w )
        cache_image[(threadIdx.y * cache_width) + threadIdx.x] = img[(y*w) + x + padding/2];

    for (int i = 0; i < levels; i ++)
    {
        int k = fb.bank[i].k;

        linear_filter lf;
        lf.hy = fy;
        lf.hx = fx;
        lf.k = fb.bank[i].k;

        int lin_index = threadIdx.y*blockDim.x + threadIdx.x;
        square_to_linear(fb.bank[i], &lf, lin_index);
        __syncthreads();
        float response = s[i][(y*w) + x] = get_filter_response_horizontal(cache_image, cache_width, cache_height, lf, threadIdx.x, threadIdx.y);
//float response = s[i][(y*w) + x] = get_filter_response_horizontal(img, w, h, lf, x, y);
//        if (x == 100 && y == 100)
//            printf("Response 100 100 horizontal is %f \n", response);
//        printf("Testing filter %d by %d \n",k,k);       
//        for (int m = 0;m < k; m++)
//            printf("%f ",lf.hx[m]);

        s[i][(y*w) + x] = response;
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
    float response = 0;

    float sk, sk_1;

    if (x >= w || y >= h)
        return;


    int k = fb.bank[0].k;

    linear_filter lf;
    lf.hx = fx;
    lf.hy = fy;

    lf.k = fb.bank[0].k;
    int lin_index = threadIdx.y*blockDim.x + threadIdx.x;
    square_to_linear(fb.bank[0], &lf, lin_index);
    // square_to_linear(fb.bank[0], &lf);
    __syncthreads();
    sk = get_filter_response_vertical(s[0], w, h, lf, x, y);
    
//    if (x == 100 && y == 100)
//        printf("Response 100 100 horizontal is %f \n", sk);

    for (int i = 0; i < levels; i ++)
    {
        k = fb.bank[i+1].k;

        linear_filter lf;
        lf.hx = fx;
        lf.hy = fy;
    
        lf.k = k;
        
        int lin_index = threadIdx.y*blockDim.x + threadIdx.x;
        square_to_linear(fb.bank[i+1], &lf, lin_index);
        // square_to_linear(fb.bank[i+1], &lf);
        __syncthreads();
        sk_1 = get_filter_response_vertical(s[i+1], w, h, lf, x, y);

        d[i][(y*w) + x] = sk - sk_1;

        sk = sk_1;
 //           if (x == 100 && y == 100)
 //       printf("Response 100 100 horizontal is %f \n", sk);

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
                            filter_bank fb)
{
    /*2D Index of current thread */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int center = 3*(1) + 1;
    const int levels = fb.n - 1;
    const float th_c = 7.0;
    const float th_r = 12.0;

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

        int idx = atomicAdd(&kp, 1);

        kp_x[idx] = x;
        kp_y[idx] = y;
  
       // dbg_printf("Keypoint detected at x = %d, y= %d. idx is %d ," 
         //           "and level is %d, and intensity is %f\n", x, y, idx, i, dk[center]);

        break;
    }

}


void DoG_detector_cuda(Mat img, int **k_x, int **k_y, int *n, float th_c, float th_r, 
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
    int block_width = BSZ;
 
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

    //DoG_Pyramid<<<grid,block>>>(gpu_img, w, h, fbd);
    vertical_blur<<<grid,block>>>(gpu_img, w, h, fbd);
    cudaDeviceSynchronize();
    end = CycleTimer::currentSeconds();
    // cout<<"Filters took "<<end-start<<" seconds"<<endl;
    horizontal_blur<<<grid,block>>>(gpu_img, w, h, fbd);
    cudaDeviceSynchronize();
    //end = CycleTimer::currentSeconds();
    //cout<<"Filters took "<<end-start<<" seconds"<<endl;
    start = CycleTimer::currentSeconds();
    DoG_Kernel<<<grid,block>>>(gpu_img, w, h, fbd);
    cudaDeviceSynchronize();
    end = CycleTimer::currentSeconds();
    //cout<<"CUDA MINMAX kernel took "<<end-start<<" seconds"<<endl;

    //cout<<"CUDA KERNEL took "<<end-start<<" seconds"<<endl;

    dbg_printf("Finished calling kernel\n");

    /* Free device memory */
    cudaFree(gpu_img);

    /* Copy results from device to host */

    cudaMemcpyFromSymbol(n, kp, sizeof(int));
    dbg_printf("Detected %d keypoints \n",*n);

    *k_y = (int *) malloc(*n * sizeof(int));
    *k_x = (int *) malloc(*n * sizeof(int));
    
    cudaMemcpyFromSymbol(*k_x, kp_x, sizeof(int)* (*n));
    cudaMemcpyFromSymbol(*k_y, kp_y, sizeof(int)* (*n));

    /* Clear kp */
    int zero = 0;
    cudaMemcpyToSymbol(kp, &zero, sizeof(int));
}


